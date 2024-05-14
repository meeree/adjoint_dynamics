# Mix memorypro and categorypro task.
import memory_pro, category_pro, mix_multi_tasks

def generate(cfg = memory_pro.DEFAULT_CFG, cfg_mix = mix_multi_tasks.MIX_DEFAULT_CFG, noise = True, debug = False):
    inp1, target1 = memory_pro.generate(cfg, debug=debug, noise=noise)
    inp2, target2 = category_pro.generate(cfg, debug=debug, noise=noise)
    inp, target = mix_multi_tasks.generate(inp1, target1, inp2, target2, cfg_mix, debug=debug, noise=noise) 
    return inp, target

def accuracy(X, Y):
    return memory_pro.accuracy(X,Y)
