program nn
    implicit none

    real,allocatable :: weights(:,:,:)
    real,allocatable :: bias(:,:), zs(:,:), as(:,:), errors(:,:), grad(:,:), tmp(:,:)
    real,allocatable :: examples(:,:), output(:,:)
    integer :: i, j, k, l, n_layers, n_examples

    n_layers = 20
    n_examples = 10
    allocate(weights(n_layers,3,3), bias(n_layers,3), zs(n_layers,3), as(n_layers,3), errors(n_layers,3))
    allocate(grad(3,3))
    allocate(examples(n_examples,3), output(n_examples,3))

    ! call random_number(examples)
    call random_number(bias)
    call random_number(weights)

    output(:,:) = exp(examples(:,:))

    do j = 1,n_examples
        do i = 1,3
            ! print*, weights(:,i,:)
        end do

        as(1,:) = examples(j,:)
        do i = 1,n_layers
            call feedforward(weights(i,:,:), bias(i,:), as(i,:), zs(i,:))
        end do

        grad(:,:) = 0
        call gradient(grad, weights, zs)
        print*, grad

        if (mod(j, 1) .EQ. 0) then
            print*, sum((as(n_layers,:) - output(j,:))**2)**(0.5)
        end if

        call output_error(output(j,:), as(n_layers,:), zs(n_layers,:), errors(n_layers,:))

        do i = 1,n_layers - 2
            errors(n_layers - i,:) = errors(n_layers - i + 1,:)
            call backpropagate(errors(n_layers - i,:), weights(n_layers - i + 1,:,:), zs(n_layers - i,:))
        end do

        do i = 2,n_layers
            call update_weights(weights(i,:,:), errors(i,:), as(i-1,:), 1.)
            call update_bias(bias(i,:), errors(i,:), 1.)
        end do
    end do

contains

    subroutine gradient(grad, weights, zs)
        real,intent(inout) :: grad(:,:)
        real,intent(in) :: weights(:,:,:), zs(:,:)
        real,allocatable :: tmp(:,:)
        integer :: n

        n = size(grad(1,:))
        allocate(tmp(n,n))

        forall (j=1:n) tmp(j,j) = sigmoid1(zs(1,j))
        do i=1,n
            grad(:,i) = matmul(tmp, weights(1,:,i))
        end do
        do l=2,n_layers
            ! all of the sigmoid matrices are diagonal, so they commute with the other matrices
            ! so they can be placed anywhere in the computation
            forall(j=1:n) grad(j,j) = grad(j,j) * sigmoid1(zs(l,j))
            grad(:,:) = matmul(grad(:,:), weights(l,:,:))
        end do
    end subroutine gradient

    ! activation of previous layer
    subroutine update_weights(weights, error, activation, mu)
        real,intent(inout) :: weights(:,:)
        real,intent(in) :: error(:), activation(:), mu
        real :: dw(3,3)
        integer :: i, j, n

        n = size(error)
        do i=1,n
            do j=1,n
                dw(i,j) = activation(j)*error(i)
            end do
        end do
        weights = weights - mu * dw
    end subroutine update_weights

    subroutine update_bias(bias, error, mu)
        real,intent(inout) :: bias(:)
        real,intent(in) :: error(:), mu
        
        bias = bias - mu * error
    end subroutine update_bias

    subroutine feedforward(weights, bias, a, z)
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

        error = matmul(weights, error)
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
