def train(model, loss_func, optimizer, X, Y, use_gpu=True, epochs=100):
    """
    :param model: torch.nn.Module
        Model
    :param loss_func: loss function
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
            if is_cuda:
                loss_value = loss.cpu().data.numpy()
            else:
                loss_value = loss.data.numpy()            
            print('\repoch = {}, loss = {:.5}'.format(epoch, loss_value))

    if is_cuda:
        model = model.cpu()

    return model