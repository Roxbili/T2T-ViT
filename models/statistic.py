import numpy as np

class MemStatistic:
    mem_eval = False
    max_forward_layer= {
        'name': '',
        'input': 0,
        'output': 0,
        'reverse': 0
    }
    reserve_stack = []

    def set_mem_eval():
        '''开启推理内存评估'''
        MemStatistic.mem_eval = True

    def reset_mem_eval():
        '''关闭推理内存评估'''
        MemStatistic.mem_eval = False

    def _check_shape(shape_list: list):
        '''检查shape的batch_size是否为1'''
        assert type(shape_list) is list

        for shape in shape_list:
            if shape[0] != 1 and shape[0] != -1:
                raise ValueError('shape[0] must be 1 or -1')
    
    def _flatten_shape_list(shape_list: list) -> list:
        '''将shape展开为size列表'''
        val = []
        for shape in shape_list:
            val.append(abs(np.prod(shape)))
        return val

    def _total_size():
        '''获得max_forward_layer所有size和'''
        return MemStatistic.max_forward_layer['input'] + MemStatistic.max_forward_layer['output'] + MemStatistic.max_forward_layer['reverse']

    def record(input_shape: list, output_shape: list, name=''):
        '''记录该操作推理内存大小，包含保留的内存'''
        MemStatistic._check_shape(input_shape)
        MemStatistic._check_shape(output_shape)

        input_size = np.sum(MemStatistic._flatten_shape_list(input_shape))
        output_size = np.sum(MemStatistic._flatten_shape_list(output_shape))
        reserve_size = np.sum(MemStatistic.reserve_stack)
        total_size = input_size + output_size + reserve_size
        if total_size > MemStatistic._total_size():
            MemStatistic.max_forward_layer['name'] = name
            MemStatistic.max_forward_layer['input'] = input_size
            MemStatistic.max_forward_layer['output'] = output_size
            MemStatistic.max_forward_layer['reserve'] = reserve_size

    def push_reserve(reserve_shape: list):
        '''记录需要暂时保留的内存空间'''
        MemStatistic._check_shape(reserve_shape)
        MemStatistic.reserve_stack.extend(MemStatistic._flatten_shape_list(reserve_shape))

    def pop_reserve():
        '''推出上一个保留的内存空间'''
        MemStatistic.reserve_stack.pop()

    def clear_reserve():
        '''清除用于skip connection的内存占用大小'''
        MemStatistic.reserve_stack.clear()

    def get_forward_info(num_type: str) -> tuple:
        '''获得该层前向推理name和峰值内存(kB)
        
            Args:
                num_type: fp32 | int8

            Return:
                name, forward_size
        '''

        forward_size = MemStatistic._total_size()
        if num_type == 'fp32':
            return (MemStatistic.max_forward_layer['name'], forward_size * 4. / 1024.)
        elif num_type == 'int8':
            return (MemStatistic.max_forward_layer['name'], forward_size * 1. / 1024.)
        else:
            raise ValueError('Invalid num_type {}'.format(num_type))