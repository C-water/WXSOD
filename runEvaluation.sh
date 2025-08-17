python Evaluation.py \
--gpu_ids='2' \
--label_path='/data0/chenquan/WX_SOD_1222/Datasets/WXSDO_data/test_real/gt/' \
--label_ext='.jpg' \
--name='WFANet' \
--test_set='real' \
--image_ext='.png'
# Evaluation.sh参数讲解
# --name：# 需要测试哪种方法的预测图？
#--gpu_ids：指定显卡编号。要先用nvidia-smi指令查看可用的GPU。一般服务器是4卡，对应的编号是0,1,2,3。根据实际情况修改。
#--test_set: 具体是哪个测试集的预测图？与label-path相关
#--label_path:哪个测试集的GT的路径？
#--test_set:具体是哪个测试集的预测图？与label-path相关
#--image_ext:预测图片的后缀？
#--label_path='./Datasets/Dataset_real_10_23/dark/mask_binary/'  \