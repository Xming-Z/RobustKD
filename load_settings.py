import time
import torch.nn as nn
import torch
import os
import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN
import models.resnet_nole as TWO_RN

def load_paper_settings(args , poison, mitigation):
    # 干净model的地址
    if not poison:
        WRN_path = os.path.join(args.data_path,'cifar', 'WRN28-4_21.09.pt')
        Pyramid_path = os.path.join(args.data_path,'cifar', 'pyramid200_mixup_15.6.pt')
    else:
        # WRN_path = os.path.join(args.data_path, 'cifar100-model','WRN28-4_maskpatch_idx_cp_ratio_0.01acc_78.15%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_maskpatch_idx_ratio_0.01acc_78.17%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_badnet_idx_ratio_0.01acc_77.61%_asr_93.21%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_sgn_idx_ratio_0.01acc_78.85%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_Poison1_clean0.5_ours_idx_acc_78.45%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'WRN28-4_Poison1_clean0.2_ours_idx_acc_78.67%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path,'cifar100-model', 'amin_batch256_clean1.0_acc_78.89%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar100-model', 'amax_batch256_clean1.0_acc_78.82%_asr_100.00%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar100-model', 'amin_batch256_clean1.0_acc_78.87%_asr_99.98%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar/teacher', 'a-lb-256_clean1.0_acc_78.66%_asr_99.75%.pt')
        # WRN_path = os.path.join(args.data_path, 'cifar/teacher', 'new0.2-CIFAR100-a_acc_78.80%_asr_92.03%.pt')
        WRN_path = os.path.join(args.data_path, 'cifar/teacher', 'new0.2-CIFAR100-a_acc_78.37%_asr_94.07%.pt')


        Pyramid_path = os.path.join(args.data_path,'cifar/teacher', 'new0.2-CIFAR100-e_acc_85.76%_asr_98.89%.pt')


    WRN_student_path = os.path.join(args.data_path, 'student_a_Mine_acc_79.32%_asr_80.62%.pt')


    if args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu', 'cuda:5': 'cuda:0'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location='cuda:0')
        teacher.load_state_dict(state)

        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        if poison:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        if poison:
            teacher.load_state_dict(state)
        else:
            teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        if poison:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']

        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        if poison:
            teacher.load_state_dict(state)
        else:
            teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)

    elif args.paper_setting == 'test-a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)

        state = torch.load(WRN_student_path, map_location={'cuda:0': 'cpu'})
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)
        student.load_state_dict(state)

    elif args.paper_setting == 'test-b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        if not poison:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        else:
            state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = TWO_RN.resnet18(num_classes=100, in_channels=3)

    elif args.paper_setting == 'test-c':
        teacher = TWO_RN.resnet50(num_classes=10, in_channels=3)
        if poison:
            # state = torch.load("/data/cifar/teacher/CIFAR10-test-c_acc_94.47%_asr_93.21%.pt")
            state = torch.load(
                "/data/cifar/teacher/new0.2-CIFAR10-test-c_acc_94.32%_asr_86.55%.pt")
        else:
            state = torch.load("/data/cifar/teacher/clean-new-resnet50_acc_9184.00%.pt")
        if mitigation:
            state = torch.load( "/data/cifar/student/CIFAR-10-mitgation-resnet50.pt")

        teacher.load_state_dict(state)

        student= TWO_RN.resnet18(num_classes=10, in_channels=3)

    elif args.paper_setting == 'test-d':

        student = TWO_RN.resnet50(num_classes=10, in_channels=3)
        state = torch.load("/data/cifar/student/CIFAR-10-mitgation-resnet50.pt")
        student.load_state_dict(state)

        teacher = student

    else:
        print('Undefined setting name !!!')


    return teacher, student, args