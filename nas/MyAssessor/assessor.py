from nni.assessor import Assessor, AssessResult
import logging

_logger = logging.getLogger(__name__)

class MemoryAssessor(Assessor):
    '''评估模型内存是否满足要求
        
        Args:
            flash_mem: 闪存可用大小
            ram_mem: 运行内存可用大小
    '''
    def __init__(self, flash_mem, ram_mem, **kwargs):
        self.flash_mem = flash_mem
        self.ram_mem = ram_mem

    def assess_trial(self, trial_job_id, trial_history):
        """
        决定 Trial 是否应该被终止。 必须重载。
        trial_history: 中间结果列表对象。
        返回 AssessResult.Good 或 AssessResult.Bad。
        """
        # return AssessResult.Bad
        return AssessResult.Good