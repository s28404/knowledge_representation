

class Trainer:
    def __init__(
        self, model, train_loader, loss_fn, optimizer
    ):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, epochs):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        history = self.model.fit(
            self.train_loader,
            epochs=epochs,
        )

        return {"avg_losses": history.history["loss"]}
