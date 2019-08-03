program nn
    implicit none

    real,dimension(3,3) :: weights
    real,dimension(3) :: a, bias
    integer :: i, j

    bias = (/ 1.0, 2.0, 3.0 /)
    call random_number(a)
    call random_number(weights)
    call feedforward(a, weights, bias)
    print*, a

contains

    subroutine feedforward(a, weights, bias)
        real,dimension(:,:),intent(in) :: weights
        real,dimension(:),intent(in) :: bias
        real,dimension(:),intent(inout) :: a
        integer :: i

        a = matmul(a, weights) + bias
        a = (/ (sigmoid(a(i)), i=1, size(a)) /)
    end subroutine feedforward

    pure function sigmoid(z) result (r)
        real,intent(in) :: z
        real :: r
        
        r = (1 + exp(-z))**(-1)
    end function sigmoid

end program nn
