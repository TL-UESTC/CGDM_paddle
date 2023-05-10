import paddle

def test_dygraph_grad(create_graph):
    x = paddle.ones(shape=[1], dtype='float32')
    x.stop_gradient = False
    y = x * x

    # Since y = x * x, dx = 2 * x
    dx = paddle.grad(
            outputs=[y],
            inputs=[x],
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True)[0]

    z = y + dx

    # If create_graph = False, the gradient of dx
    # would not be backpropagated. Therefore,
    # z = x * x + dx, and x.gradient() = 2 * x = 2.0

    # If create_graph = True, the gradient of dx
    # would be backpropagated. Therefore,
    # z = x * x + dx = x * x + 2 * x, and
    # x.gradient() = 2 * x + 2 = 4.0

    z.backward()
    return x.gradient()

print(test_dygraph_grad(create_graph=False)) # [2.]
print(test_dygraph_grad(create_graph=True)) # [4.]