
module Models

export sample_mps_to_file, cluster_ising, quantum_ising, quantum_XYZ,process_model

using ITensors, ITensorMPS

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

#MODELS:


function cluster_ising_z(N::Int,W::Int,h::Float64)
  os = OpSum()
  for j = 0:N-3
     os += -8.0,"Sx",j+1,"Sz",(j+1)%N+1,"Sx",(j+2)%N+1
     os += h*0.004, "Sy",j+1,"Sy",(j+1)%N+1
  end
  os+=h*0.004,"Sy",N-1,"Sy",N
  return os
end


function cluster_ising_x(N::Int,W::Int,h::Float64)
  os = OpSum()
  for j = 0:N-3
     os += -8.0,"Sz",j+1,"Sx",(j+1)%N+1,"Sz",(j+2)%N+1
     os += h*0.004, "Sy",j+1,"Sy",(j+1)%N+1
  end
  os+=h*0.004,"Sy",N-1,"Sy",N
  return os
end



function cluster_ising_y(N::Int,W::Int,h::Float64)
  os = OpSum()
  for j = 0:N-3
     os += 8.0,"Sx",j+1,"Sy",(j+1)%N+1,"Sx",(j+2)%N+1
     os += h*0.004, "Sz",j+1,"Sz",(j+1)%N+1
  end
  os+=h*0.004,"Sz",N-1,"Sz",N
  return os
end




function quantum_ising(N::Int,W::Int,h::Float64)
  os = OpSum()
  if W>1
   for i = 0:W-2
     for j = 0:N-1
      os += -4.0,"Sz",j+1+i*N, "Sz",(j+1)%N+1+i*N
      os += -4.0,"Sz",j+1+i*N, "Sz",j+1+((i+1)%W)*N
      os += -h*0.002, "Sx", j+1+i*W
     end
   end
  else
   for j = 0:N-1
     os += -4.0,"Sz",j+1, "Sz",(j+1)%N+1
     os += -h*0.002, "Sx",j+1
   end
  end
  return os
end

function quantum_XYZ(N::Int,W::Int,h::Float64)
  os = OpSum()
  if W>1
   for i = 0:W-1
     for j = 0:N-1
      os += -0.04*h,"Sz",j+1+i*W, "Sz",(j+1)%N+1+i*N
      os += -0.04*h,"Sz",j+1+i*W, "Sz",j+1+((i+1)%W)*N
      os += -4.0,"Sx",j+1+i*W, "Sx",(j+1)%N+1+i*N
      os += -4.0,"Sx",j+1+i*W, "Sx",j+1+((i+1)%W)*N
      os += -4.0,"Sy",j+1+i*W, "Sy",(j+1)%N+1+i*N
      os += -4.0,"Sy",j+1+i*W, "Sy",j+1+((i+1)%W)*N
     end
   end
   
  else
   for j = 0:N-1
     os += -0.04*h,"Sz",j+1, "Sz",(j+1)%N+1
     os += -4.0,"Sx",j+1, "Sx",(j+1)%N+1
     os += -4.0,"Sy",j+1, "Sy",(j+1)%N+1
   end
  end
  return os
end

function process_model(model::String,N::Int,W::Int,h::Float64)
    if model == "XYZ"
       return quantum_XYZ(N,W,h)
    elseif model == "CIM_X"
       return cluster_ising_x(N,W,h)
    elseif model == "CIM_Y"
       return cluster_ising_y(N,W,h)
    elseif model == "CIM_Z"
       return cluster_ising_z(N,W,h)
    elseif model == "QIM"
       return quantum_ising(N,W,h)
    else
	error("Invalid model type: $model. Accepted models are XYZ CIM_(X/Y/Z) QIM")
    end	       

end

end