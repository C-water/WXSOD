python Train.py \
--name='WFANet' \
--image_train_dir='/data0/chenquan/WX_SOD_1222/Datasets/WXSDO_data/train_sys/input/' \
--image_train_ext='.jpg' \
--label_train_dir='/data0/chenquan/WX_SOD_1222/Datasets/WXSDO_data/train_sys/gt/' \
--label_train_ext='.jpg' \
--image_test_dir='/data0/chenquan/WX_SOD_1222/Datasets/WXSDO_data/test_sys/input/' \
--image_test_ext='.jpg' \
--label_test_dir='/data0/chenquan/WX_SOD_1222/Datasets/WXSDO_data/test_sys/gt/' \
--label_test_ext='.jpg' \
--epochs=52 \
--batchsize=6 \
--image_size=384 \
--lr=0.0001 \
--gpu_ids='2'

# run.sh参数讲解
#--name：保存的模型名称，需要写详细，我们约定一个命名规则：0707_FSMINet_384_e200_b8
#--pth：网络模型最优的模型权重。每次训练都会保存很多的权重，我们需要在log文件里对比定量结果，来筛选出我们期望的权重。
#--save_path：输出图片的保存路径。'/results/'(固定的！！！) +  ‘0707_FSMINet_384_e200_b8’
#--image_size：输入图片的尺寸，和训练保持一致！
#--test_set：测试集的名称。因此，实际的图片保存位置为  '/results/'(固定的！！！) + '0707_FSMINet_384_e200_b8' + '/' + 'EORSSD'
#--image_test_dir：训练（测试）需要用到的输入图和标签的路径
#--image_test_ext：用于定义数据集图片的后缀，需要结合实际情况来修改。常见的后缀有'.jpg'、'.png'
#--gpu_ids：指定显卡编号。要先用nvidia-smi指令查看可用的GPU。一般服务器是4卡，对应的编号是0,1,2,3。根据实际情况修改。