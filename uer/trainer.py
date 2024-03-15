from ast import arg
import time
import torch
import sys
import copy
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.logging import init_logger
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed

from torch.utils.tensorboard import SummaryWriter

def train_and_validate(args):
    set_seed(args.seed)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path)
        if True:  # ! for train
            for n, p in list(model.named_parameters()):
                if "compactor" in n:
                    torch.nn.init.eye_(p.data)
    else:
        for n, p in list(model.named_parameters()):
            if "compactor" in n:
                torch.nn.init.eye_(p.data)
            elif "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.deepspeed:
        worker(args.local_rank, None, args, model)
    elif args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)

class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.l2_lambda = args.l2_lambda
        self.labels_num = args.labels_num

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.logger = args.logger
        self.logger_tfb = args.logger_tfb

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()

        mask_dict = {}
        bias_mask_dict = {}
        loader_iter = iter(loader)
        caculate_mask_list = []
        if args.mask_path is not None:
            f = open(args.mask_path,'w',5)
        for name, parms in model.named_parameters():
            if "bias" in name:
                bias_mask_dict[name] = None
            if "compactor" in name:
                mask_dict[name] = None
                if "encoder.transformer.0.self_attn.left_compactor.2" in name or "self_attn.final_linear_left_compactor" in name \
                         or "feed_forward.linear_2_left_compactor" in name \
                         or "target.mlm_linear_2_left_compactor" in name:
                    caculate_mask_list.append(name)
        mask_dict_clone = copy.deepcopy(mask_dict)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model,mask_dict,self.current_step)
            if args.deepspeed:
                model.backward(loss)
            else:
                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if args.weight_squeeze:
                if self.current_step == args.w_step or self.current_step % args.w_step == 0:
                    mask_dict = copy.deepcopy(mask_dict_clone)
                    for name, parms in model.named_parameters():
                        if name in caculate_mask_list:
                            parms_norm = torch.linalg.norm(parms.data, dim=0)
                            dim = int(self.current_step / args.w_step)*((args.hidden_size - args.target_dim)/16)
                            if dim > (args.hidden_size - args.target_dim):
                                dim = args.hidden_size - args.target_dim
                            if "feed_forward.linear_2_left_compactor" in name:
                                dim = dim * 4
                            if "self_attn.left_compactor.2" in name or "self_attn.final_linear_left_compactor" in name:
                                if args.compactor_mask_strategy == "dim":
                                    chunk_list = list(torch.chunk(parms_norm, int(args.hidden_size/64)))
                                    parms_mask_l = []
                                    indices_l = []
                                    cut_dim_per_head = int((args.hidden_size - args.target_dim)/12)
                                    cut_dim_now = int(self.current_step / args.w_step)*int(cut_dim_per_head/8)
                                    if cut_dim_now*12 > (args.hidden_size - args.target_dim):
                                        cut_dim_now = cut_dim_per_head
                                    for i,data in enumerate(chunk_list):
                                        _, indices = data.topk(int(cut_dim_now), largest=False)
                                        for nums in indices.tolist():
                                            indices_l.append(nums+64*i)
                                        parms_mask_l.append(data > data[indices[-1]])
                                    for i,parms_mask_head in enumerate(parms_mask_l):
                                        if i == 0:
                                            parms_mask = parms_mask_head
                                        else:
                                            parms_mask = torch.cat([parms_mask,parms_mask_head])
                                    indices =indices_l
                                else:
                                    chunk_list = list(torch.chunk(parms_norm, int(args.hidden_size/64)))
                                    for i,data in enumerate(chunk_list):
                                        chunk_list[i] = torch.sum(data)
                                    chunk_list = torch.tensor(chunk_list).to(parms.data.device)
                                    head_num = int(self.current_step / (2*args.w_step))
                                    if head_num>=int((args.hidden_size - args.target_dim)/64):
                                        head_num = int((args.hidden_size - args.target_dim)/64)
                                    _, indices = chunk_list.topk(head_num,largest=False)
                                    parms_mask = torch.ones(args.hidden_size).to(parms.data.device)
                                    for k in indices:
                                        parms_mask[k*64:(k+1)*64] = 0
                                    parms_mask = parms_mask > 0
                            else:
                                _, indices = parms_norm.topk(int(dim), largest=False)
                                parms_mask = parms_norm > parms_norm[indices[-1]]
                                if args.compactor_mask_strategy == "dim":
                                    indices = indices.tolist()
                            mask_dict[name] = parms_mask
                            if self.current_step % 10000 == 0 and (not self.dist_train or (self.dist_train and rank == 0)):
                                if args.compactor_mask_strategy == "dim":
                                    f.write(str(self.current_step)+'\t'+str(name)+'\t'+str(indices)+'\n')
                                else:
                                    f.write(str(self.current_step)+'\t'+str(name)+'\t'+str(indices.tolist())+'\n')
                        elif "self_attn.left_compactor.2" in name or "target.mlm_linear_1_left_compactor" in name or "feed_forward.linear_1_left_compactor" in name:
                            mask_dict[name] = mask_dict["module.encoder.transformer.0.self_attn.left_compactor.2.weight"]
                    temp = []
                    if self.dist_train:
                        prefix = "module."
                    else:
                        prefix = ""
                    for name, parms in model.named_parameters():
                        if name in mask_dict and mask_dict[name] == None:
                            temp.append(name)
                        elif name in mask_dict and mask_dict[name] != None:
                            for temp_name in temp:
                                mask_dict[temp_name] = mask_dict[name]
                            temp = []
                    bias_mask_dict[prefix+"embedding.layer_norm.gamma"] = \
                        mask_dict[prefix+"embedding.emb_compactor.weight"]
                    bias_mask_dict[prefix+"embedding.layer_norm.beta"] = \
                        mask_dict[prefix+"embedding.emb_compactor.weight"]
                    for i in range(args.layers_num):
                        bias_mask_dict[prefix+"encoder.transformer.{}.self_attn.linear_layers.0.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.right_compactor.2.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.self_attn.linear_layers.1.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.right_compactor.2.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.self_attn.linear_layers.2.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.right_compactor.2.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.layer_norm_1.gamma".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.layer_norm_1.beta".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(i)]
                        
                        bias_mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.self_attn.final_linear_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_1.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_1_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_2.bias".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_2_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.layer_norm_2.gamma".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_2_right_compactor.weight".format(i)]
                        bias_mask_dict[prefix+"encoder.transformer.{}.layer_norm_2.beta".format(i)] = \
                        mask_dict[prefix+"encoder.transformer.{}.feed_forward.linear_2_right_compactor.weight".format(i)]

                    bias_mask_dict[prefix+"target.mlm_linear_1.bias"] = \
                    mask_dict[prefix+"target.mlm_linear_1_right_compactor.weight".format(i)]
                    bias_mask_dict[prefix+"target.layer_norm.gamma"] = \
                    mask_dict[prefix+"target.mlm_linear_1_right_compactor.weight".format(i)]
                    bias_mask_dict[prefix+"target.layer_norm.beta"] = \
                    mask_dict[prefix+"target.mlm_linear_1_right_compactor.weight".format(i)]
                for name, parms in model.named_parameters():
                    if ("bias" in name or "gamma" in name or "beta" in name ) and self.current_step > args.w_step and "mlm_linear_2" not in name:
                        parms_mask = bias_mask_dict[name]
                        parms.grad.data = torch.where(parms_mask, parms.grad.data, min(args.l2_lambda,1.0)*parms.data)
                        if self.l2_lambda > 1 :
                            penalty_grad2 = torch.where(parms_mask, torch.zeros_like(parms.data), (args.l2_lambda - 1)*parms.data)
                            optimizer.state[parms]['exp_avg'].mul_(0.9).add_(penalty_grad2, alpha=0.1)
                    if "compactor" in name:
                        if "left" in name:
                            parms_norm = torch.linalg.norm(parms.data, dim=0)
                            penalty_grad = parms.data # L2 grad
                        else:
                            parms_norm = torch.linalg.norm(parms.data, dim=1)
                            penalty_grad = parms.data
                        if self.current_step > args.w_step:
                            N = parms.data.size(0)
                            parms_mask = mask_dict[name]

                            if "left" in name:
                                parms_mask = parms_mask.unsqueeze(0).repeat(N, 1)
                            else:
                                parms_mask = parms_mask.unsqueeze(1).repeat(1, N)
                            if self.current_step > args.w_step:
                                parms.grad.data = torch.where(parms_mask, parms.grad.data, torch.zeros_like(penalty_grad))
                        dim_h = int(args.hidden_size - args.target_dim)

                        if self.current_step > args.w_step:
                            parms.grad.data = torch.where(parms_mask, parms.grad.data, min(args.l2_lambda,1.0)*penalty_grad)
                            if self.l2_lambda > 1 :
                                penalty_grad2 = torch.where(parms_mask, torch.zeros_like(penalty_grad), (args.l2_lambda - 1)*penalty_grad)
                                optimizer.state[parms]['exp_avg'].mul_(0.9).add_(penalty_grad2, alpha=0.1)

                        else:
                            parms.grad.data.add_(penalty_grad)
                        
                        # ! tensorboard
                        if args.print_ins_log and ("0.self_attn.right_compactor.0" in name or "0.self_attn.linear_layers.0.bias" in name) and \
                                self.current_step % 100 == 0 and (not self.dist_train or (self.dist_train and rank == 0)):
                            self.logger.info("norm:{}".format(parms_norm[0:5]))
                            self.logger.info("penalty grad:{}".format(penalty_grad[0][0:10]))
                            self.logger.info("origin grad:{}".format(parms.grad.data[0][0:10]))
                            if self.current_step> args.w_step :
                                self.logger.info("mask:{}".format(mask_dict[name][0:10]))
                            tag = name.replace('.', '/')
                            self.logger_tfb.add_histogram(tag + "/norm", parms_norm, self.current_step)


            if self.current_step % self.accumulation_steps == 0:
                if args.deepspeed:
                    model.step()
                else:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()


            if args.deepspeed:
                if self.current_step % self.save_checkpoint_steps == 0:
                    model.save_checkpoint(self.output_model_path, str(self.current_step))
            else:
                if self.current_step % self.save_checkpoint_steps == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model,mask_dict,current_step):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg,mask_dict,current_step,soft_tgt=None)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
                         "| {:8.2f} tokens/s"
                         "| loss {:7.2f}"
                         "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        # ! write to tensorboard
        self.logger_tfb.add_scalar('loss', self.total_loss / self.report_steps, self.current_step)
        self.logger_tfb.add_scalar('acc', self.total_correct / self.total_denominator, self.current_step)      

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0




str2trainer = {"mlm": MlmTrainer}


def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    # Get logger
    args.logger = init_logger(args)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend=args.backend)
        rank = dist.get_rank()
        gpu_id = proc_id
    elif args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, rank,
                                                           args.world_size, True)
    else:
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, 0, 1, True)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]

    p1 = []
    p2 = []
    p3 = []
    for n, p in param_optimizer:
        if "compactor" in n:
            p1.append(p)
        else:
            if not any(nd in n for nd in no_decay):
                p2.append(p)
            else:
                p3.append(p)
        if args.comp_only:
            print("compactor only!")
            optimizer_grouped_parameters = []
        else:
            optimizer_grouped_parameters = [
                {'params': p2, "weight_decay": 0.01},
                {"params": p3, "weight_decay": 0.0}
            ]

        if len(p1) > 0:
            optimizer_grouped_parameters.append(
                {'params': p1, "lr": args.compactor_learning_rate, "weight_decay": 0.01})

    if args.optimizer in ["adamw"]:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                         correct_bias=False)
    else:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                         scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps * args.warmup)
    else:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps * args.warmup,
                                                         args.total_steps)

    if args.deepspeed:
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=optimizer_grouped_parameters,
            args=args,
            optimizer=custom_optimizer,
            lr_scheduler=custom_scheduler,
            mpu=None,
            dist_init_required=False)
    else:
        if gpu_id is not None:
            model.cuda(gpu_id)
        optimizer = custom_optimizer
        scheduler = custom_scheduler
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp

        if args.dist_train:
            # Initialize multiprocessing distributed training environment.
            dist.init_process_group(backend=args.backend,
                                    init_method=args.master_ip,
                                    world_size=args.world_size,
                                    rank=rank)
            model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            args.logger.info("Worker %d is training ... " % rank)
        else:
            args.logger.info("Worker is training ...")
    # print all param
    if rank == None or rank == 0:
        args.logger.info("all params: ")
        for key, value in args.__dict__.items():
            if key in ["logger", "tokenizer", "vocab"]:
                continue
            args.logger.info("  - {}: {}".format(key, value))
    # ! tenserboard
    args.logger_tfb = SummaryWriter(log_dir=args.tb_path)
    trainer = str2trainer[args.data_processor](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
