
class EarlyStopping:
    def __init__(self,patience = 15,min_delta=0,mode = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode 
        self.count = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self,score):
        if self.best_score is None:
            self.best_score = score
            return False 
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.count += 1
            else:
                self.best_score = score 
                self.count = 0
        else:
            if score > self.best_score - self.min_delta:
                self.count += 1
            else:
                self.best_score = score 
                self.count = 0
        if self.count >= self.patience:
            self.early_stop = True 
        return self.early_stop 