program nn
    implicit none

    real,allocatable :: weights(:,:,:)
    real,allocatable :: bias(:,:), zs(:,:), as(:,:), errors(:,:)
    real,dimension(3) :: example
    integer :: i, n_layers

    n_layers = 10
    allocate(weights(n_layers,3,3), bias(n_layers,3), zs(n_layers,3), as(n_layers,3), errors(n_layers,3))

    do i = 1,n_layers
        bias(i,:) = (/ 1, 2, 3 /)
    end do

    call random_number(as(1,:))
    ! call random_number(bias)
    call random_number(weights)

    do i = 1,n_layers
        call feedforward(as(i,:), weights(i,:,:), bias(i,:), zs(i,:))
    end do


    example = (/ 40, 50, 60 /)
    call output_error(example, as(n_layers,:), zs(n_layers,:), errors(1,:))

    do i = 1,n_layers - 1
        errors(i + 1,:) = errors(i,:)
        call backpropagate(errors(i+1,:), weights(n_layers - i,:,:), zs(n_layers - i,:))
    end do

    do i = 1,n_layers
        call update_weights(weights(i,:,:), errors(i,:), as(i-1,:), 0.01)
        call update_bias(bias(i,:), errors(i,:), 0.01)
    end do

contains

    ! activation of previous layer
    subroutine update_weights(weights, error, activation, mu)
        real,intent(inout) :: weights(:,:)
        real,intent(in) :: error(:), activation(:), mu

        weights = weights - mu * dot_product(error, activation)
    end subroutine update_weights

    subroutine update_bias(bias, error, mu)
        real,intent(inout) :: bias(:)
        real,intent(in) :: error(:), mu
        
        bias = bias - mu * sum(error)
    end subroutine update_bias

    subroutine feedforward(a, weights, bias, z)
        real,intent(in) :: weights(:,:), bias(:)
        real,dimension(:),intent(inout) :: a, z
        integer :: i

        z = matmul(a, weights) + bias
        a = (/ (sigmoid(z(i)), i=1, size(a)) /)
    end subroutine feedforward

    ! computes the "error" in the output layer
    ! given by d^L_j = \frac{\delta C}{\delta a^L_j} \sigma'(z^L_j)
    subroutine output_error(example, output, zs, error)
        real,dimension(:),intent(in) :: example, output, zs
        real,dimension(:),intent(out) :: error

        error = (/ (output(i) - example(i), i=1, size(error)) /)
        error = (/ (error(i) * sigmoid1(zs(i)), i=1, size(error)) /)
    end subroutine output_error

    subroutine backpropagate(error, weights, zs)
        real,dimension(:),intent(inout) :: error
        real,intent(in) :: weights(:,:), zs(:)

        error = matmul(transpose(weights), error)
        error = (/ (error(i) * sigmoid1(zs(i)), i=1, size(error)) /)
    end subroutine backpropagate

    pure function sigmoid(z) result(r)
        real,intent(in) :: z
        real :: r
        
        r = (1 + exp(-z))**(-1)
    end function sigmoid

    pure function sigmoid1(z) result(r)
        real,intent(in) :: z
        real :: r

        r = sigmoid(-z) * (1 - sigmoid(-z))
    end function sigmoid1

end program nn
