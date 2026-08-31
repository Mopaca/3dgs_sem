import math

def get_priority_top_percent(iteration, start_iter=5000, end_iter=15000,
                             start_percent=0.02, end_percent=0.0015):
    """
    Exponential percentile schedule:
      - at start_iter: start_percent
      - at end_iter:   end_percent

    Default:
      start_iter=5000,  start_percent=0.02   (2%)
      end_iter=15000,   end_percent=0.0015   (0.15%)
    """

    if iteration <= start_iter:
        return start_percent
    if iteration >= end_iter:
        return end_percent

    progress = (iteration - start_iter) / float(end_iter - start_iter)

    # p(t) = start_percent * (end_percent / start_percent) ^ progress
    top_percent = start_percent * ((end_percent / start_percent) ** progress)
    return top_percent


def get_dynamic_priority_top_percent(self, current_iteration, densify_until_iter, percentile_ref, adc_grad_threshold, start_percent=0.02, end_percent=0.0015):
    if self.priority_decay_start_iter is None:
        if  float(percentile_ref.item()) <= float(adc_grad_threshold):
            self.priority_decay_start_iter = int(current_iteration)
    
    if self.priority_decay_start_iter is None:
        return start_percent
    
    if current_iteration >= densify_until_iter:
        return end_percent
    
    decay_span = max(1, densify_until_iter - self.priority_decay_start_iter)
    progress = (current_iteration - self.priority.decay_start_iter) / float(decay_span)
    progress = max(0.0, min(1.0, progress))

    current_top_percent = start_percent * ((end_percent / start_percent) ** progress)
    return current_top_percent