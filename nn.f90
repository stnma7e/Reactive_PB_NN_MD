program main
    implicit none

    real,dimension(3,3) :: weights
    real,dimension(3) :: bias, output
    integer :: i, j

    bias = (/ 1.0, 2.0, 3.0 /)
    call random_number(weights)
    print*, feedforward(weights, bias)

contains

    function feedforward(weights, bias) result(z)
        real,dimension(3,3),intent(in) :: weights
        real,dimension(3),intent(in) :: bias
        real,dimension(3) :: z

        z = matmul(bias, weights) 
    end function feedforward

end program main
