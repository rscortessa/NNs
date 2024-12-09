using ITensors, ITensorMPS

let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  J = parse(Float64,ARGS[2])
  h = parse(Float64,ARGS[3])
  NS = parse(Int,ARGS[4])
  sites = siteinds("S=1/2", N*2)
  println("L=$N","J=$J","G=$h","NS=$NS")
  # Define the Hamiltonian for the 1D Heisenberg model
  os = OpSum()

  for j = 0:N-1
  
      os += -4.0,"Sz",j+1, "Sz",(j+1)%N+1
      os += -4.0,"Sz",j+1+N, "Sz",(j+1)%N+1+N
     
      os += -J*0.04,"Sz",j+1, "Sz",j+1+N
      
      os += -h*0.02, "Sx", j+1
      os += -h*0.02, "Sx", j+1+N
      

  end
  
  H = MPO(os, sites)

  # Initialize a random MPS
  psi0 = random_mps(sites)

  # Run DMRG to find the ground state
  nsweeps = N
  maxdim = [10*i for i in 1:nsweeps]

  cutoff = 1E-10
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  println("Final energy = $energy")

# Function to sample MPS and store configurations in a file
function sample_mps_to_file(psi::MPS, filename::String, N::Int)
    # Open a file for writing
    open(filename, "w") do file
        for _ in 1:N
            # Sample a configuration from the MPS
            config = sample!(psi)
	    # config.-= 3/2
	    # config.*=2
	    config.=round.(Int,(config.-3/2)*2)
            # Convert the configuration into a string
            # The configuration `config` is a vector of spins at each site
            config_str = join(string.(config), " ")  # Convert each spin to a string
            println(file, config_str)  # Write to file
        end
    end
    println("Sampling complete. Configurations saved to $filename")
end

filename = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "MPSG" * ARGS[3]* ".txt"  # Output file to store configurations

# Call the function to sample and save to the file
sample_mps_to_file(psi, filename, NS)
  nothing
  
end
