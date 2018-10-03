import numpy as np
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = FLAGS.DEFAULT_IN + 'celebA_NCHW/'
from GeneralTools.graph_func import Agent
from DeepLearning.my_sngan2 import SNGan

num_file = 9
filename = ['celebA_{:03d}'.format(i) for i in range(num_file)]
act_k = True  # multiplier
w_nm = 's'  # spectral normalization
architecture = {'input': [(3, 64, 64)],
                'code': [(128, 'linear')],
                'generator': [{'name': 'l1', 'out': 1024 * 4 * 4, 'op': 'd', 'act': 'linear', 'act_nm': None,
                               'out_reshape': [1024, 4, 4]},
                              {'name': 'l2_up', 'out': 512, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4,
                               'strides': 2},
                              {'name': 'l3_up', 'out': 256, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4,
                               'strides': 2},
                              {'name': 'l4_up', 'out': 128, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4,
                               'strides': 2},
                              {'name': 'l5_up', 'out': 64, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4,
                               'strides': 2},
                              {'name': 'l6_t32', 'out': 3, 'act': 'tanh'}],
                'discriminator': [{'name': 'l1_f32', 'out': 64, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
                                  {'name': 'l2_ds', 'out': 128, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm,
                                   'kernel': 4, 'strides': 2},
                                  {'name': 'l3', 'out': 128, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
                                  {'name': 'l4_ds', 'out': 256, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm,
                                   'kernel': 4, 'strides': 2},
                                  {'name': 'l5', 'out': 256, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
                                  {'name': 'l6_ds', 'out': 512, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm,
                                   'kernel': 4, 'strides': 2},
                                  {'name': 'l7', 'out': 512, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
                                  {'name': 'l8_ds', 'out': 1024, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm,
                                   'kernel': 4, 'strides': 2},
                                  {'name': 'l9', 'out': 1024, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm,
                                   'out_reshape': [1024 * 4 * 4]},
                                  {'name': 'l10_s', 'out': 16, 'op': 'd', 'act_k': act_k, 'w_nm': w_nm}]}

debug_mode = False
optimizer = 'adam'
num_instance = 22511*num_file
save_per_step = 12500
batch_size = 64
num_class = 0
end_lr = 1e-7
num_threads = 7

# random code to test model
code_x = np.random.randn(400, 128).astype(np.float32)
# to show the model improvements over iterations, consider save the random codes and use later
# np.savetxt('MMD-GAN/z_128.txt', z_batch, fmt='%.6f', delimiter=',')
# code_x = np.genfromtxt('MMD-GAN/z_128.txt', delimiter=',', dtype=np.float32)

# case 1
lr_list = [2e-4, 2e-4]  # [dis, gen]
loss_type = 'logistic'
sample_same_class = False
sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])

agent = Agent(
    filename, sub_folder, load_ckpt=True, do_trace=False,
    do_save=True, debug_mode=debug_mode, debug_step=400,
    query_step=1000, log_device=False, imbalanced_update=None,
    print_loss=True)

mdl = SNGan(
    architecture, num_class=num_class, loss_type=loss_type,
    optimizer=optimizer, do_summary=True, do_summary_image=True,
    num_summary_image=8, image_transpose=False)

# for i in range(1):
#     # mdl.training(
#     #     filename, agent, num_instance, lr_list, end_lr=end_lr, max_step=save_per_step,
#     #     batch_size=batch_size, sample_same_class=sample_same_class, num_threads=num_threads)
#     # if debug_mode is not None:
#     #     _ = mdl.eval_sampling(
#     #         filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
#     #         real_sample=False, do_embedding=False, do_sprite=True)
#     if debug_mode is False:
#         scores = mdl.mdl_score(
#             filename, sub_folder, batch_size, num_batch=781, model='v1')
#         print('Epoch {} with scores: {}'.format(i, scores))

# ms-ssim
# import time
#
# start_time = time.time()
# scores = mdl.mdl_score(
#     filename, sub_folder, batch_size, num_batch=781, model='ms_ssim', ckpt_file=None)
# print('MS-SSIM scores: {}'.format(scores))
# print('\n Calculation took {:.1f} seconds'.format(time.time() - start_time))

# image generation
# iter_list = ['12500', '25000', '37500', '50000', '62500', '75000', '87500', '100000',
#              '112500', '125000', '137500', '150000', '162500', '175000', '187500', '200000']
# iter_list = iter_list[7:8]
# print(iter_list)
# for i in range(len(iter_list)):
#     _ = mdl.eval_sampling(
#         filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
#         real_sample=False, do_embedding=False, do_sprite=True)
_ = mdl.eval_sampling(
    filename, sub_folder, mesh_num=(6, 6), mesh_mode=0, #code_x=code_x,
    real_sample=False, do_embedding=False, do_sprite=True)

# case 1
print('Chunk of code finished.')
