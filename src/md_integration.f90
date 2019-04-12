
module md_integration
  use routines
  implicit none

contains

  !*************************************************************************
  !  this subroutine controls what ensemble to run, and calls appropriate subroutines
  !  currently, only NVE molecular dynamics is implemented
  !
  !  data structure molecule_data will be changed if MS-EVB simulation, and we
  !  have proton hop, hence intent(inout)
  !
  !  verlet_list will be changed in energy routines if needs update
  !*************************************************************************

  subroutine mc_sample( system_data , molecule_data , atom_data, integrator_data, verlet_list_data, PME_data, file_io_data )
    use global_variables
    type( system_data_type ) , intent(inout)    :: system_data
    type( molecule_data_type ), dimension(:), intent(inout) :: molecule_data
    type( atom_data_type )  , intent(inout)   :: atom_data
    type( integrator_data_type ) , intent(in) :: integrator_data
    type(verlet_list_data_type) , intent(inout)  :: verlet_list_data
    type(PME_data_type)     , intent(inout)      :: PME_data
    type(file_io_data_type) , intent(in)         :: file_io_data

    ! print*, trajectory_step

    Select Case( integrator_data%ensemble )
    Case("NPT")
        if (modulo(trajectory_step, system_data%barofreq) == 0) then
            call monte_carlo_barostat(system_data, molecule_data, atom_data, integrator_data, verlet_list_data, PME_data, file_io_data)
        endif
    End Select

    call md_integrate_atomic(system_data, molecule_data, atom_data, integrator_data, verlet_list_data, PME_data, file_io_data)


  end subroutine mc_sample



  !*************************************************************************
  ! samples velocities for all atoms from Maxwell-Boltzmann
  ! units of velocities are Angstrom/ps 
  !
  ! we allow the possibility of freezing atoms during a simulation, which is
  ! flagged using the atype_freeze data structure.  These frozen atoms will
  ! have zero velocity and won't contribute to the temperature
  !*************************************************************************
  subroutine sample_atomic_velocities(n_mole, n_atom, temperature, molecule_data, atom_data )
    use global_variables
    integer, intent(in)          :: n_mole,n_atom
    real*8,   intent(in)         :: temperature
    type(molecule_data_type), dimension(:),intent(in) :: molecule_data
    type(atom_data_type) , intent(inout)   :: atom_data

    ! allocate temporary arrays here for convenience
    real*8,dimension(:), allocatable :: mass
    integer,dimension(:), allocatable :: atom_type_index
     
    integer :: i_atom, i_type, n_tot
    real*8,dimension(2)  :: vel
    real*8,parameter :: small=1D-3
    real*8           :: conv_fac, kB
    real*8 :: sum_KE, norm

    conv_fac = constants%conv_kJmol_ang2ps2gmol   ! converts kJ/mol to A^2/ps^2*g/mol
    kB       = constants%boltzmann                ! 0.008314 kJ/mol/K

    allocate(mass(n_atom), atom_type_index(n_atom) )
    mass            = atom_data%mass
    atom_type_index = atom_data%atom_type_index
    
    n_tot=0
    sum_KE=0d0

    ! zero velocity, in case we're freezing an atom
    atom_data%velocity=0d0

    !**************** first pull velocities from maxwell-boltzmann distribution
       do i_atom=1, n_atom
          ! make sure mass is non-zero
          if ( mass(i_atom) < small ) then
             write(*,*) "trying to assign velocity for atom ", i_atom
             write(*,*) "but mass is zero!"
             stop
          end if

          i_type = atom_type_index(i_atom)
          ! get velocity if atomtype isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             ! gaussian random numbers come in 2
             call max_boltz(vel,mass(i_atom),temperature, kB)
             atom_data%velocity(1,i_atom)=vel(1)
             atom_data%velocity(2,i_atom)=vel(2)
             call max_boltz(vel,mass(i_atom),temperature, kB)
             atom_data%velocity(3,i_atom)=vel(1)
          end if

       enddo

    ! get rid of excess center of mass momentum of system
    call subtract_center_of_mass_momentum(n_mole, molecule_data, atom_data )

    ! now rescale velocities to desired temperature
       do i_atom=1, n_atom
          i_type = atom_type_index(i_atom)
          ! add KE if atom isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             n_tot = n_tot + 1
             sum_KE = sum_KE + 0.5d0 * mass(i_atom) * dot_product(atom_data%velocity(:,i_atom),atom_data%velocity(:,i_atom)) / conv_fac
          end if
       enddo

    norm = 1.5d0 * kB * temperature * dble(n_tot) / sum_KE
    atom_data%velocity = atom_data%velocity * sqrt(norm)

    deallocate(mass, atom_type_index )

  end subroutine sample_atomic_velocities




  !*******************************************************
  ! this subroutine calculates the center of mass momentum of the system,
  ! and subtracts the total net per atom contribution from each atom's momentum,
  ! so that the net COM momentum is zero
  !*******************************************************
  subroutine subtract_center_of_mass_momentum(n_mole, molecule_data, atom_data )
    use global_variables
    integer,intent(in)::n_mole
    type(molecule_data_type) , dimension(:), intent(in) :: molecule_data
    type(atom_data_type) , intent(inout)  :: atom_data

    !******** this is a local data structure with pointers that will be set
    ! to subarrays of atom_data arrays for the specific atoms in the molecule
    type(single_molecule_data_type) :: single_molecule_data


    integer :: i_mole, i_atom, i_type, n_tot
    real*8,dimension(3) :: rho_system, rho_excess

    n_tot=0
    rho_system=0d0

    !**************** calculate total COM momentum
    do i_mole=1,n_mole

       ! set pointers for this data structure to target molecule
       call return_molecule_block( single_molecule_data , molecule_data(i_mole)%n_atom, molecule_data(i_mole)%atom_index, atom_velocity=atom_data%velocity, atom_mass=atom_data%mass, atom_type_index=atom_data%atom_type_index )

       do i_atom=1, molecule_data(i_mole)%n_atom
          i_type = single_molecule_data%atom_type_index(i_atom)
          ! add momentum if atomtype isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             n_tot = n_tot + 1
             rho_system(:) = rho_system(:) + single_molecule_data%mass(i_atom) * single_molecule_data%velocity(:,i_atom)
          end if
       enddo
    enddo

    !*************  now subtract excess momentum from each atom, so that center of mass momentum is zero
    rho_excess(:) = rho_system(:) / dble(n_tot)

    do i_mole=1,n_mole

       ! set pointers for this data structure to target molecule
       call return_molecule_block( single_molecule_data , molecule_data(i_mole)%n_atom, molecule_data(i_mole)%atom_index, atom_velocity=atom_data%velocity, atom_mass=atom_data%mass, atom_type_index=atom_data%atom_type_index )

       do i_atom=1, molecule_data(i_mole)%n_atom
          i_type = single_molecule_data%atom_type_index(i_atom)     
          ! change velocity if atom isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             single_molecule_data%velocity(:,i_atom) = single_molecule_data%velocity(:,i_atom) - rho_excess(:) / single_molecule_data%mass(i_atom)
          end if
       enddo
    enddo

  call dissociate_single_molecule_data(single_molecule_data)

  end subroutine subtract_center_of_mass_momentum

  !************************************************************************
  ! This is a Langevin Integrator used for NVT simulations. This is used in
  ! conjunction with the md_integrate_atomic subroutine shown below. This
  ! uses Leapfrog Langevin Integration, following the OpenMM implementation.
  ! The friction coefficient used here is equal to 1/ps.
  !************************************************************************
  subroutine langevin_integrator( system_data, atom_data, i_atom, conv_fac, dt )
    use global_variables
    type( system_data_type ), intent(inout)     :: system_data
    type( atom_data_type ) , intent(inout)      :: atom_data
    integer, intent(in)                         :: i_atom
    real*8, intent(in)                          :: conv_fac
    real*8, intent(in)                          :: dt

    real*8 :: f_coeff
    real*8 :: kb
    real*8 :: temperature
    real*8 :: zr
    real*8, dimension(2) :: z
    real*8, dimension(3) :: rand

    f_coeff = constants%friction_coeff
    kb = constants%boltzmann
    temperature = system_data%temperature
    ! Get a random number for all 3 velocity components
    do
       call random_number(z)
       z=2.0*z-1.0
       zr=z(1)**2+z(2)**2
       if (zr > 0.0 .and. zr < 1.0) exit
    end do
    zr=sqrt(-2.0*log(zr)/zr)
    z=z*zr
    rand(1) = z(1)
    rand(2) = z(2)
    do
       call random_number(z)
       z=2.0*z-1.0
       zr=z(1)**2+z(2)**2
       if (zr > 0.0 .and. zr < 1.0) exit
    end do
    zr=sqrt(-2.0*log(zr)/zr)
    z=z*zr
    rand(3) = z(1)

    atom_data%velocity(:,i_atom) = Exp(-1*f_coeff*dt/2d0) * atom_data%velocity(:,i_atom) + (1-Exp(-1*f_coeff*dt/2d0))/f_coeff *atom_data%force(:,i_atom) / atom_data%mass(i_atom) * conv_fac + sqrt(2*kb*temperature*f_coeff*1000d0) / sqrt(atom_data%mass(i_atom)/1000d0) / 100d0 * sqrt((1-Exp(-1*f_coeff*dt))/(2d0*f_coeff)) * rand(:)

    end subroutine langevin_integrator


  subroutine monte_carlo_barostat(system_data, molecule_data, atom_data, integrator_data, verlet_list_data, PME_data, file_io_data )
      use global_variables
      use total_energy_forces
      use ms_evb
      type( system_data_type ), intent(inout)     :: system_data
      type( molecule_data_type ), dimension(:), intent(inout)   :: molecule_data
      type( atom_data_type ) , intent(inout)      :: atom_data
      type( integrator_data_type ), intent(in)    :: integrator_data
      type(verlet_list_data_type), intent(inout)  :: verlet_list_data
      type(PME_data_type)     , intent(inout)     :: PME_data
      type(file_io_data_type) , intent(in)        :: file_io_data

      integer, save :: n_trials, n_accept
      logical :: accepted

      ! storage for data in case move is rejected
      real*8, dimension(3,10000) :: positions
      real*8, dimension(3,10000) :: saved_forces
      real*8, dimension(6) :: saved_energies

      integer :: i,j, verlet_flag_junk, h3o_index
      real*8 :: box_vec, dbox_vec
      real*8 :: deltalen, oldboxlen, newboxlen, Vi
      real*8 :: kT, Eold, Enew, w, rand, pV, S
      real*8, parameter :: conv = 6.022/10**5 ! 10**5 * 10**-3 * 10**-30 * 6.022*10**23 ! bar to Pa to kJ/m^3 to kJ/A^3 to kJ/mol/A^3

      ! check if box is cubic; if not we crashin out
      box_vec = dot_product(system_data%box(:,1), system_data%box(:,1))
      do i=2,3
        dbox_vec = box_vec - dot_product(system_data%box(:,i), system_data%box(:,i))
        if (dbox_vec .GE. 0.001) error stop "monte carlo barostat cannot be used with non-cubic box"
      end do

      n_trials = n_trials + 1

      kT = constants%boltzmann * system_data%temperature ! kJ/mol
      Eold = system_data%potential_energy ! kJ/mol
      Vi = system_data%volume

      ! save pre-move configuation
      do i=1,system_data%total_atoms
        positions(:,i) = atom_data%xyz(:,i)
      end do

      do i=1,system_data%total_atoms
        saved_forces(:,i) = atom_data%force(:,i)
      end do

      saved_energies(1) = system_data%potential_energy
      saved_energies(2) = system_data%E_elec
      saved_energies(3) = system_data%E_vdw
      saved_energies(4) = system_data%E_bond
      saved_energies(5) = system_data%E_angle
      saved_energies(6) = system_data%E_dihedral

      ! make new periodic box
      oldboxlen = system_data%box(1,1) ! this okay because cubic => all elements of box(i, i!=j) are 0
      call random_number(rand)
      deltalen = system_data%box(1,1) * system_data%baroscale * (rand * 2 - 1)
      newboxlen = oldboxlen + deltalen
      do j=1,3
          system_data%box(j,j) = system_data%box(j,j) + deltalen
      end do
      call periodic_box_change(system_data, PME_data, verlet_list_data, atom_data, molecule_data)

      ! scale molecular coordinates to new box size
      do i=1,system_data%n_mole
        call scale_coordinates(i, newboxlen/oldboxlen, system_data, molecule_data, atom_data)
      end do

      h3o_index = hydronium_molecule_index(1)

      !************** get energy ****************!
          Select Case(ms_evb_simulation)
          Case("yes")
              call ms_evb_calculate_total_force_energy( system_data, molecule_data, atom_data, verlet_list_data, PME_data, file_io_data, integrator_data%n_output )
          Case("no")
              call calculate_total_force_energy(system_data, molecule_data, atom_data, verlet_list_data, PME_data)
          End Select
      !*****************************************!


      !**************** test our monte carlo move *************************!
      Enew = system_data%potential_energy

      pV = conv * system_data%pressure * (system_data%volume - Vi)
      S = system_data.n_mole * kT * 3*log(newboxlen/oldboxlen)
      w = Enew-Eold + pV - S

      if ( h3o_index /= hydronium_molecule_index(1) ) then
          print *, "volume move coincident with proton transfer"
          w = -1 ! force the volume move to accept so that proton jumps don't
                 ! break the code
      end if

      accepted = .true.
      if (w >= 0) then
          call random_number(rand)
          if (rand > exp(-w/kT)) then
              ! move has been rejected
              ! undo all of the changes that were made to the system
              do j=1,3
                  system_data%box(j,j) = system_data%box(j,j) - deltalen
              end do
              call periodic_box_change(system_data, PME_data, verlet_list_data, atom_data, molecule_data)

              do i=1,system_data%total_atoms
                atom_data%xyz(:,i) = positions(:,i)
              end do

              do i=1,system_data%total_atoms
                atom_data%force(:,i) = saved_forces(:,i)
              end do

              system_data%potential_energy = saved_energies(1)
              system_data%E_elec     = saved_energies(2)
              system_data%E_vdw      = saved_energies(3)
              system_data%E_bond     = saved_energies(4)
              system_data%E_angle    = saved_energies(5)
              system_data%E_dihedral = saved_energies(6)

              accepted = .false.
          endif
      endif

      if (accepted) then
          n_accept = n_accept + 1
          verlet_list_data%flag_verlet_list = 1

          print*, "step", trajectory_step, "volume", system_data%volume, "density", system_data%n_mole / system_data%volume
      endif


      if(debug .eq. 1) then
          if (accepted) then
              write(*,*), "monte carlo move accepted"
          else
              write(*,*), "monte carlo move rejected"
          endif
          write(*,*), "cubic box length", system_data%box(1,1)
          write(*,*), "dlen", deltalen
          write(*,*), "monte carlo metropolis terms: (Enew - Eold) pV S w"
          write(*,*), Enew-Eold, pV, S, w
          write(*,*), "end monte carlo move"
      endif

      if (n_trials > 10) then
          if (n_accept < 0.25*n_trials) then
              system_data%baroscale = system_data%baroscale / 1.1
              n_trials = 0
              n_accept = 0
          else if (n_accept > 0.75*n_trials) then
              system_data%baroscale = system_data%baroscale * 1.1
              n_trials = 0
              n_accept = 0
          endif
      endif

  end subroutine monte_carlo_barostat

  subroutine scale_coordinates(mol_index, boxscale, system_data, molecule_data, atom_data)
      use global_variables
      type( system_data_type ), intent(inout)                   :: system_data
      type( molecule_data_type ), dimension(:), intent(inout)   :: molecule_data
      type( atom_data_type ) , intent(inout)                    :: atom_data
      integer, intent(in) :: mol_index
      real*8, intent(in) :: boxscale

      integer :: j, k
      type(molecule_data_type) :: mol

      ! vector for center of mass in box coordinates
      real*8, dimension(3) :: r_com_box

      ! displacement vector from the center of mass for each atom in the molecule
      ! num atoms is fixed to length 10 so that it can be stack allocated
      real*8, dimension(10,3) :: dr_com

      mol = molecule_data(mol_index)
      ! find displacement vectors for each atom
      do j=1,mol%n_atom
          dr_com(j, :) =  mol%r_com(:) - atom_data%xyz(:, mol%atom_index(j))
      end do

      ! scale the center of mass vector using box coordinates
      ! r_com_box = matmul(system_data%xyz_to_box_transform, mol%r_com)
      ! mol%r_com = matmul(system_data%box, r_com_box(:) * boxscale)
      mol%r_com(:) = mol%r_com(:) * boxscale ! because the box is cubic

      ! update atomic positions for the molecule using previously calculated
      ! displacements
      do j=1,mol%n_atom
          atom_data%xyz(:, mol%atom_index(j)) = mol%r_com(:) - dr_com(j, :)
      end do
  end subroutine


  !************************************************************************
  ! this is the MD engine for atomistic molecular simulations.  
  ! currently, this uses the velocity verlet algorithm to integrate Newton's equations
  !  velocity units are A/ps
  !
  ! NOTE we have already checked for non-zero atomic masses in the sample_atomic_velocities
  ! subroutine, and so here we don't worry about it
  !
  !  data structure molecule_data will be changed if MS-EVB simulation, and we
  !  have proton hop, hence intent(inout)
  !
  !  verlet_list will be changed in energy routines if needs update
  !
  !************************************************************************
  subroutine md_integrate_atomic( system_data , molecule_data , atom_data, integrator_data, verlet_list_data, PME_data, file_io_data )
    use global_variables
    use total_energy_forces
    use ms_evb
    type( system_data_type ), intent(inout)     :: system_data
    type( molecule_data_type ), dimension(:), intent(inout)   :: molecule_data
    type( atom_data_type ) , intent(inout)      :: atom_data
    type( integrator_data_type ), intent(in)    :: integrator_data
    type(verlet_list_data_type), intent(inout)  :: verlet_list_data
    type(PME_data_type)     , intent(inout)     :: PME_data
    type(file_io_data_type) , intent(in)        :: file_io_data

    integer :: i_atom, i_type, total_atoms, h3o_index, i, j
    real*8  :: dt , conv_fac
    integer, dimension(4) :: atom_num

    integer, parameter :: max_neighbors = 10000
    integer :: n_neighbors
    type(atom_data_type), save :: h3o_neighborhood
    ! check if one of the arrays in the data structure has been allocated
    ! if not, then this is probably the first iteration of this function
    if (.not. allocated(h3o_neighborhood%xyz)) call allocate_atom_data(h3o_neighborhood, max_neighbors + 4)

    ! save the neighborhood around the hydronium in case a proton transfer
    ! occurs
    call save_hydronium_neighborhood( system_data , molecule_data , atom_data, verlet_list_data, file_io_data, h3o_neighborhood, n_neighbors )


    !****************************timing**************************************!
    if(debug .eq. 1) then
       call date_and_time(date,time)
       write(*,*) "md integration step started at", time
    endif
    !***********************************************************************!

    ! define local variables for convenience
    dt = integrator_data%delta_t
    conv_fac = constants%conv_kJmol_ang2ps2gmol  ! converts kJ/mol to A^2/ps^2*g/mol
    total_atoms = system_data%total_atoms

    h3o_index = hydronium_molecule_index(1)
    !atom_num = molecule_data(h3o_index)%atom_index

    !if ( mod( trajectory_step, integrator_data%n_output ) == 0 ) then
    !do i=atom_num(1), atom_num(4)
    !    write(file_io_data%ofile_hamiltonian_file_h,*) atom_data%velocity(:, i)
    !enddo
    !write(file_io_data%ofile_hamiltonian_file_h,*) ""    
   ! endif


    !******************* Velocity Verlet Integrator or Langevin Leapfrog 

    ! first calculate velocities at delta_t / 2
       do i_atom = 1, total_atoms
          i_type = atom_data%atom_type_index(i_atom)

          ! update position, velocity if atom isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             ! first calculate velocities at delta_t / 2
             ! here force is in kJ/mol*Ang^-1
             Select Case( integrator_data%ensemble )
             Case("NVE")
             atom_data%velocity(:,i_atom) = atom_data%velocity(:,i_atom) + dt / 2d0 / atom_data%mass(i_atom) * atom_data%force(:,i_atom) * conv_fac
             Case Default
             call langevin_integrator( system_data, atom_data, i_atom, conv_fac, dt )
             End Select

             ! now calculate new atomic coordinates at delta_t
             atom_data%xyz(:,i_atom) = atom_data%xyz(:,i_atom) + atom_data%velocity(:,i_atom) * dt
          end if
       end do

       ! after updating atomic coordinates, calculate new center of mass of molecules
       call update_r_com( system_data%n_mole, molecule_data, atom_data )
       ! translate molecules back into the box if they have left
       call shift_molecules_into_box( system_data%n_mole , molecule_data , atom_data , system_data%box,  system_data%xyz_to_box_transform )


    !**********************get total forces and energies*****************************!
    Select Case(ms_evb_simulation)
    Case("yes")
       call ms_evb_calculate_total_force_energy( system_data, molecule_data, atom_data, verlet_list_data, PME_data, file_io_data, integrator_data%n_output )
    Case("no")
       call calculate_total_force_energy( system_data, molecule_data, atom_data, verlet_list_data, PME_data )
    End Select
    !********************************************************************************!

    ! print hydronium neighborhood if the proton jumped
    if (h3o_index /= hydronium_molecule_index(1)) then
        write (*,*) "BEGIN neighborhood positions BEFORE JUMP"
        call print_neighborhood_data(h3o_neighborhood, n_neighbors)
        write(*,*), "BEGIN neighborhood positions AFTER JUMP"
        call save_hydronium_neighborhood( system_data , molecule_data , atom_data, verlet_list_data, file_io_data, h3o_neighborhood, n_neighbors )
        call print_neighborhood_data(h3o_neighborhood, n_neighbors)
        write(*,*) "END neighborhood positions"
    end if

     ! now final velocities
    do i_atom = 1, total_atoms
        i_type = atom_data%atom_type_index(i_atom)

          ! update position, velocity if atom isn't frozen
          if ( atype_freeze(i_type) /= 1 ) then
             ! here force is in kJ/mol*Ang^-1
             Select Case( integrator_data%ensemble )
             Case("NVE")
             atom_data%velocity(:,i_atom) = atom_data%velocity(:,i_atom) + dt / 2d0 / atom_data%mass(i_atom) * atom_data%force(:,i_atom) * conv_fac
             Case Default
             call langevin_integrator( system_data, atom_data, i_atom, conv_fac, dt )
             End Select


             ! make sure forces aren't crazy
             if ( ( abs( atom_data%force(1,i_atom) ) > 10d4 ) .or. ( abs( atom_data%force(2,i_atom) ) > 10d4 ) .or. ( abs( atom_data%force(3,i_atom) ) > 10d4 ) ) then
                write(*,*) "force on atom ", i_atom , " is too big ", atom_data%force(:,i_atom)
                stop
             end if

          end if
    end do


    ! finally remove center of mass momentum.  This should be numerical noise, so shouldn't effect energy conservation
    call subtract_center_of_mass_momentum(system_data%n_mole, molecule_data, atom_data )

    !****************************timing**************************************!
    if(debug .eq. 1) then
       call date_and_time(date,time)
       write(*,*) "md integration step finished at", time
    endif
    !***********************************************************************!

  end subroutine md_integrate_atomic

  subroutine save_hydronium_neighborhood( system_data , molecule_data , atom_data, verlet_list_data, file_io_data, neighborhood_data, n_neighbors )
    use global_variables
    type( system_data_type ), intent(in)     :: system_data
    type( molecule_data_type ), dimension(:), intent(in)   :: molecule_data
    type( atom_data_type ) , intent(in)      :: atom_data
    type(verlet_list_data_type), intent(in)  :: verlet_list_data
    type(file_io_data_type) , intent(in)     :: file_io_data
    type( atom_data_type ) , intent(in)      :: neighborhood_data
    integer, intent (out) :: n_neighbors

    integer :: i_atom, i_type, total_atoms, h3o_index, i, j, pos_index
    integer :: oxygen_index, verlet_start, verlet_finish
    ! this will be used to store the previous frame's atomic_positions
    ! which will be printed when a proton hops

    h3o_index = hydronium_molecule_index(1)

    pos_index = 5
    do i = 1,4
        neighborhood_data%xyz(:,i)             = atom_data%xyz(:,h3o_index)
        neighborhood_data%velocity(:,i)        = atom_data%velocity(:,h3o_index)
        neighborhood_data%force(:,i)           = atom_data%force(:,h3o_index)
        neighborhood_data%atom_type_index(i) = atom_data%atom_type_index(h3o_index)
    end do

    oxygen_index = molecule_data(h3o_index)%atom_index(1)
    verlet_start = verlet_list_data%verlet_point(oxygen_index)
    verlet_finish = verlet_list_data%verlet_point(oxygen_index + 1) - 1
    n_neighbors = verlet_finish - verlet_start + 1

    ! find all neighbor indices of the oxygen atom
    do i=verlet_start, verlet_finish
        i_atom = verlet_list_data%neighbor_list(i)
        neighborhood_data%xyz(:,pos_index)           = atom_data%xyz(:,i_atom)
        neighborhood_data%velocity(:,pos_index)      = atom_data%velocity(:,i_atom)
        neighborhood_data%force(:,pos_index)         = atom_data%force(:,i_atom)
        neighborhood_data%atom_type_index(pos_index) = atom_data%atom_type_index(i_atom)
        neighborhood_data%mass(pos_index)            = i_atom
        pos_index = pos_index + 1
    end do

    end subroutine save_hydronium_neighborhood

    subroutine allocate_atom_data( atom_data, n_atoms )
        use global_variables
        type(atom_data_type), intent(inout) :: atom_data
        integer, intent(in) :: n_atoms

        allocate( atom_data%xyz(3,n_atoms) )
        allocate( atom_data%velocity(3,n_atoms) )
        allocate( atom_data%force(3,n_atoms) )
        allocate( atom_data%atom_type_index(n_atoms) )
        allocate( atom_data%mass(n_atoms) ) ! this will be used for storing real atom index
    end subroutine allocate_atom_data

    subroutine print_neighborhood_data(neighborhood_data, n_neighbors) 
        use global_variables
        type(atom_data_type), intent(in) :: neighborhood_data
        integer, intent(in) :: n_neighbors

        integer :: i, i_type, i_atom

        write(*,*), "BEGIN hydronium position; step:", trajectory_step
        do i=1,n_neighbors + 4
            if (i == 5) then
                write (*, '(A, I6, A, I)'), "BEGIN hydronium neighborhood (", n_neighbors, " atoms); step:", trajectory_step
            end if

            i_atom = neighborhood_data%mass(i)
            i_type = neighborhood_data%atom_type_index(i)
            write(*, '(A, I, F10.4, F10.4, F10.4)'), atype_name(i_type), i_atom, neighborhood_data%xyz(:,i)
        end do
    end subroutine print_neighborhood_data

end module md_integration
