from replay_buffer import ReplayBuffer

buffer_size = 10 
loop_size = 20 

memory = ReplayBuffer(buffer_size, input_shape=1, n_action=1)

for i in range(loop_size):
    memory.store_transitions(i, i, i, i, i)
    
print("Testing to ensure the first state memory is correct ")
assert memory.state_mem[0] == 9

print("Test Successful")