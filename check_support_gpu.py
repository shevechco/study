#!/usr/bin/python3
#coding:utf-8
import argparse

def test_tensorflow():
    import tensorflow as tf
    print("Tensorflow Version：{}".format(tf.__version__))
    print("Support GPU：{}".format(tf.test.is_gpu_available()))
    ''''高级版本不支持list_physical_devices 方法'''
    try:
        cpus = tf.config.list_physical_devices(device_type="GPU")
        gpus = tf.config.list_physical_devices(device_type="CPU")
        print(cpus)
        print(gpus)
    except Exception as e:
        print("Current version not support list_physical_devices function !")
        
def test_prtorch():
    import torch
    print("PyTorch Version: {}".format(torch.__version__))
    print("Support GPU：{}".format(torch.cuda.is_available()))
    ngpu= 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Current used {}".format(device))
    try:
        print(torch.cuda.get_device_name(0))
        print(torch.rand(3,3).cuda())
    except Exception as e:
        print("Current version not support GPU !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "AI框架GPU检测支持脚本 !")
    parser.add_argument("--f", default="tensorflow", required=True, help="输入测试的AI框架名称，tensorflow 或 pytorch")
    args = parser.parse_args()
    if args.f.lower() == "tensorflow":
        test_tensorflow()
    if args.f.lower() == "torch" or args.f.lower() == "pytorch":
        test_prtorch()
