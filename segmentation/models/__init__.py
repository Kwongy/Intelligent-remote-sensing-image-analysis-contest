import torchvision.models as models

from models.swiftnet import swiftnet


def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s', 'lafcn8s']:
        model = model(n_classes=n_classes)
        # initNetParams(model)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == 'unet':
        model = model(n_channels=3, n_classes=n_classes)
    elif name == 'deeplab_vgg':
        model = model()
        # model.init_parameters()
    elif name == 'DFCN':
        model = model(n_classes=n_classes)
        # initNetParams(model)
    elif name == 'fpn_resnet50':
        model = model(n_classes=n_classes)
    elif name == 'fpn101':
        model = model(n_classes=n_classes)
        # initNetParams(model)
    elif name == 'line_fpn_resnet50':
        model = model(n_classes=n_classes)
    elif name == 'stacked_fpn101':
        model = model(n_classes=n_classes)
    elif name == 'surper_fpn':
        model = model(n_classes=n_classes)
    elif name == 'hed':
        model = model(n_classes=n_classes)
    elif name == 'ESFPN_v1':
        model = model(n_classes=n_classes)
    elif name == 'swiftnet':
        model = model(num_classes=n_classes)
    elif name == 'easwiftnet':
        model = model(num_classes=n_classes)
    elif name == 'easwiftnet':
        model = model(num_classes=n_classes)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'fpn101': fpn101,
        'line_fpn_resnet50': line_fpn_resnet50,
        'stacked_fpn101': stacked_fpn101,
        'surper_fpn': surper_fpn,
        'ESFPN_v1': ESFPN_v1,
        'swiftnet': swiftnet,
        'easwiftnet': easwiftnet,
        # 'hed': hed,
    }[name]


def initNetParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def load_pretrained_model(model, model_dir):
    model_dict = model.state_dict()
    weight = torch.load(model_dir)
    pretrained_dict = weight  # .state_dict()
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict.pop('final.weight')
    pretrained_dict.pop('final.bias')
    # for k, v in pretrained_dict.items():
    #     name = k[7:]  # remove `module.`
    #     if name[:4] == 'aspp':
    #         continue
    #     elif name[:11] == 'score_fused':
    #         continue
    #     new_state_dict[name] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(model_dir, 'loaded')
    return model
