def check_shape(name,tensor,expectedshape):
        assert tensor.shape == expectedshape, f" expected shape {expectedshape}, {name} shape: {tensor.shape}"

