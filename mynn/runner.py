import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    Modified to calculate loss and metrics once per epoch instead of every iteration
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 1)
        save_dir = kwargs.get("save_dir", "best_model")
        patience = kwargs.get("patience", 10)  # 获取早停参数

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        no_improve_count = 0  # 早停计数器

        for epoch in range(num_epochs):
            # Training phase
            X, y = train_set
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            total_train_loss = 0.0
            total_train_score = 0.0
            num_batches = 0

            n_samples = X.shape[0]
            n_iterations = (n_samples + self.batch_size - 1) // self.batch_size

            for iteration in range(n_iterations):
                start = iteration * self.batch_size
                end = start + self.batch_size
                if end > n_samples:
                    end = n_samples
                train_X = X[start:end]
                train_y = y[start:end]

                # Forward pass
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)

                # Accumulate metrics
                total_train_loss += trn_loss
                total_train_score += trn_score
                num_batches += 1

                # Backward pass and optimize
                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            # Calculate epoch metrics
            avg_train_loss = total_train_loss / num_batches
            avg_train_score = total_train_score / num_batches
            self.train_loss.append(avg_train_loss)
            self.train_scores.append(avg_train_score)

            # Evaluation phase
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            # Early stopping logic
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                best_score = dev_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}, no improvement for {patience} consecutive epochs.")
                    break  # 提前终止训练循环

            # Logging
            if (epoch + 1) % log_epochs == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}")
                print(f"[Train] Loss: {avg_train_loss:.4f}, Score: {avg_train_score:.4f}")
                print(f"[Dev]   Loss: {dev_loss:.4f}, Score: {dev_score:.4f}\n")

            # 如果触发早停，需要跳出外层循环
            if no_improve_count >= patience:
                break

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)




# import numpy as np
# import os
# from tqdm import tqdm

# class RunnerM():
#     """
#     This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
#     due to the different implementation of those models.
#     """
#     def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.metric = metric
#         self.scheduler = scheduler
#         self.batch_size = batch_size

#         self.train_scores = []
#         self.dev_scores = []
#         self.train_loss = []
#         self.dev_loss = []

#     def train(self, train_set, dev_set, **kwargs):

#         num_epochs = kwargs.get("num_epochs", 0)
#         log_iters = kwargs.get("log_iters", 100)
#         save_dir = kwargs.get("save_dir", "best_model")

#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)

#         best_score = 0

#         for epoch in range(num_epochs):
#             # self.model.train_mode()
#             X, y = train_set

#             assert X.shape[0] == y.shape[0]

#             idx = np.random.permutation(range(X.shape[0]))

#             X = X[idx]
#             y = y[idx]

#             for iteration in range(int(X.shape[0] / self.batch_size) + 1):
#                 train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
#                 train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

#                 logits = self.model(train_X)
#                 trn_loss = self.loss_fn(logits, train_y)
#                 self.train_loss.append(trn_loss)
                
#                 trn_score = self.metric(logits, train_y)
#                 self.train_scores.append(trn_score)

#                 # the loss_fn layer will propagate the gradients.
#                 self.loss_fn.backward()
#                 # loss_grad = self.loss_fn.backward()
#                 # self.model.backward(loss_grad)

#                 self.optimizer.step()
#                 if self.scheduler is not None:
#                     self.scheduler.step()
                
#                 # self.model.eval_mode()
#                 dev_score, dev_loss = self.evaluate(dev_set)
#                 self.dev_scores.append(dev_score)
#                 self.dev_loss.append(dev_loss)
#                 # self.model.train_mode()

#                 # if (iteration) % (log_iters) == 0:
#                 print(f"epoch: {epoch}, iteration: {iteration}")
#                 print(f"[Train] loss: {trn_loss}, score: {trn_score}")
#                 print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

#             if dev_score > best_score:
#                 save_path = os.path.join(save_dir, 'best_model.pickle')
#                 self.save_model(save_path)
#                 # print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
#                 best_score = dev_score
#         self.best_score = best_score

#     def evaluate(self, data_set):
#         # self.model.eval_mode()
#         X, y = data_set
#         logits = self.model(X)
#         loss = self.loss_fn(logits, y)
#         score = self.metric(logits, y)
#         return score, loss
    
#     def save_model(self, save_path):
#         self.model.save_model(save_path)