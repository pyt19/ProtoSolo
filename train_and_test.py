import time
import torch
from torch.nn.modules.loss import BCELoss
from utils.helpers import list_of_distances, make_one_hot
import sklearn.metrics as metrics
import torch.nn as nn

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    import time
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0

    # separation cost is meaningful only for class_specific

    total_separation_cost = 0
    total_avg_separation_cost = 0
    criteon = nn.CrossEntropyLoss().cuda()

    for i, (image, label) in enumerate(dataloader):
        image = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            output, min_distances = model(image)
            cross_entropy = criteon(output, target)

            if class_specific:

                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()

                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

                cluster_cost = torch.mean(max_dist - inverted_distances)
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)

                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity_s).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)

                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)

            n_examples += target.size(0)

            n_correct += (predicted == target).sum().item()

            n_batches += 1

            total_cross_entropy += cross_entropy.item()

            total_cluster_cost += cluster_cost.item()

            total_separation_cost += separation_cost.item()

            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del image
        del target
        del output
        del predicted
        del min_distances

    end = time.time()
    time = end - start
    cross_ent = total_cross_entropy / n_batches
    cluster = total_cluster_cost / n_batches
    separation = total_separation_cost / n_batches
    avg_separation = total_avg_separation_cost / n_batches
    accu = n_correct / n_examples * 100
    l1 = model.module.last_layer.weight.norm(p=1).item()

    log('\ttime: \t{0}'.format(time))

    log('\tcross ent: \t{0}'.format(cross_ent))

    log('\tcluster: \t{0}'.format(cluster))

    if class_specific:
        log('\tseparation:\t{0}'.format(separation))
        log('\tavg separation:\t{0}'.format(avg_separation))

    log('\taccu: \t\t{0}%'.format(accu))

    log('\tl1: \t\t{0}'.format(l1))

    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()

    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    p_dist_pair = p_avg_pair_dist.item()
    log('\tp dist pair: \t{0}'.format(p_dist_pair))

    accuracy = n_correct / n_examples
    result = []
    result.append(time)
    result.append(cross_ent)
    result.append(cluster)
    result.append(separation)
    result.append(avg_separation)
    result.append(accu)
    result.append(l1)
    result.append(p_dist_pair)

    return accuracy, result

def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert (optimizer is not None)
    log('\ttrain')
    model.train()
    accuracy, train_result = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                                            class_specific=class_specific, coefs=coefs, log=log)
    return accuracy, train_result

def tst(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    accuracy, test_result = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                                           class_specific=class_specific, log=log)
    return accuracy, test_result


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False

    model.module.prototype_vectors.requires_grad = False

    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True

    model.module.prototype_vectors.requires_grad = True

    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\twarm')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True

    model.module.prototype_vectors.requires_grad = True

    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tjoint')
