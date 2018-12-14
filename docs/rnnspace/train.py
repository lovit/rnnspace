import time


def train(model, loss_function, optimizer, X, Y, use_gpu=True, epochs=100):
    """
    :param model: torch.nn.Module
        Model
    :param loss_function: loss function
    :param optimizer: optimizer
    :param X: list of torch.Tensor
        List of input sequences
    :param Y: list of torch.Tensor
        List of output sequences
    :param use_gpu: Boolean
        If True, use CUDA. Default is True
    :param epochs: int
        Maximum number of epochs

    It returns
    ----------
    model : torch.nn.Module
        Trained model
    """

    is_cuda = use_gpu and torch.cuda.is_available()

    if is_cuda:
        print('CUDA is available')
        model = model.cuda()

    for epoch in range(epochs):

        t = time.time()

        for i, (chars, tags) in enumerate(zip(X, Y)):
            if is_cuda:
                chars = chars.cuda()
                tags = tags.cuda()
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model(chars)
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()

            if (i % 1000 == 0):
                print('\repoch = {}, iter = {}'.format(epoch, i), end='')

        if (epoch < 10) or (epoch % 100 == 0):
            t = time.time() - t
            if is_cuda:
                loss_value = loss.cpu().data.numpy()
            else:
                loss_value = loss.data.numpy()
            loss_value = float(loss_value)
            print('\repoch = {}, loss = {:.5}, time={}'.format(
                epoch, loss_value, time.strftime('%H:%M:%S', time.gmtime(t))))

    if is_cuda:
        model = model.cpu()

    return model