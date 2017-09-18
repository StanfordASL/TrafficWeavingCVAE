trapz(x::Vector, dt, x0=0) = cumsum(vcat(0, 0.5*dt*(x[1:end-1] + x[2:end]))) + x0
trapz(x::Matrix, dt, x0=0) = cumsum(hcat(zeros(size(x,1)), 0.5*dt*(x[:,1:end-1] + x[:,2:end])), 2) .+ x0
vel(xd, yd) = sqrt.(xd.^2 + yd.^2)
acc(xd, xdd, yd, ydd, v=vel(xd,yd)) = (xd.*xdd + yd.*ydd)./v
curv(xd, xdd, yd, ydd, v=vel(xd,yd)) = (xd.*ydd - xdd.*yd)./v.^3

@inbounds @views function all_future_xy(xdd_choices, y_targets, state0, tibvp,
                                        a0 = [(1,1)], ply=4, t_hold=3, dt=Float32(0.1))
    Nx = length(xdd_choices)
    Ny = length(y_targets)
    if a0 isa Vector{Tuple{Int,Int}}
        X = zeros(Float32, Nx^ply, 1 + length(a0) + ply*t_hold, 3)
        Y = zeros(Float32, Ny^ply, 1 + length(a0) + ply*t_hold, 3)
        X[:,1,:] .= state0[1:2:end]'
        Y[:,1,:] .= state0[2:2:end]'
        t0 = 1 + length(a0)
        for (t, (xi, yi)) in enumerate(a0)
            X[:,t+1,3] .= xdd_choices[xi]
            q0 = SVector(Y[1,t,1],
                         Y[1,t,2],
                         Y[1,t,3])
            qf = SVector(y_targets[yi], 0f0, 0f0)
            c, u = tibvp(q0, qf, 100f0)
            if dt > duration(u)
                q = qf
            else
                q = u.x(q0, u.y, u.t, dt)
            end
            Y[:,t+1,1] .= q[1]
            Y[:,t+1,2] .= q[2]
            Y[:,t+1,3] .= q[3]
        end
    elseif a0 isa Tuple
        X0, Y0 = a0
        X0 = reshape(X0, (1, size(X0)...))
        Y0 = reshape(Y0, (1, size(Y0)...))
        X = zeros(Float32, Nx^ply, 1 + size(X0,2) + ply*t_hold, 3)
        Y = zeros(Float32, Ny^ply, 1 + size(Y0,2) + ply*t_hold, 3)
        X[:,1,:] .= state0[1:2:end]'
        Y[:,1,:] .= state0[2:2:end]'
        X[:,2:1+size(X0,2),:] .= X0
        Y[:,2:1+size(Y0,2),:] .= Y0
        t0 = 1 + size(X0,2)
    else
        error("Incompatible a0 type.")
    end

    # xdd, xd, x
    for (r, ci) in enumerate(CartesianRange(Tuple(Nx for i in 1:ply)))
        for (p, v) in enumerate(ci.I)
            for t in 1:t_hold
                X[r,t0+t_hold*(p-1)+t,3] = xdd_choices[v]
            end
        end
    end
    cumsum!(X[:,2:end,2], X[:,1:end-1,3]*dt, 2)
    X[:,2:end,2] .+= X[:,1,2]
    cumsum!(X[:,2:end,1], (X[:,1:end-1,2] .+ X[:,2:end,2])*dt./2, 2)
    X[:,2:end,1] .+= X[:,1,1]

    # ydd, yd, y
    for (r, ci) in enumerate(CartesianRange(Tuple(Ny for i in 1:ply)))
        for (p, v) in enumerate(ci.I)
            curr_step = t0 + (p-1)*t_hold
            q0 = SVector(Y[r,curr_step,1],
                         Y[r,curr_step,2],
                         Y[r,curr_step,3])
            qf = SVector(y_targets[v], 0f0, 0f0)
            c, u = tibvp(q0, qf, 100f0)
            for t in 1:t_hold
                if dt*t > duration(u)
                    q = qf
                else
                    q = u.x(q0, u.y, u.t, t*dt)
                end
                Y[r,curr_step+t,1] = q[1]
                Y[r,curr_step+t,2] = q[2]
                Y[r,curr_step+t,3] = q[3]
            end
        end
    end
    X, Y
end

function crossover_time(car1::Matrix, car2::Matrix)
    dx = car1[:,1:1] - car2[:,1:1]
    dv = car1[:,3:3] - car2[:,3:3]
    -dx./dv
end

function crossover_time(car1::Array{T,3}, car2::Array{S,3}) where {T,S}
    dx = car1[:,:,1:1] - car2[:,:,1:1]
    dv = car1[:,:,3:3] - car2[:,:,3:3]
    -dx./dv
end

function extra_features(car1::Matrix, car2::Matrix, column_list=String[])
    extras_tensor = zeros(Float32, size(car1, 1), length(column_list))
    for (i, c) in enumerate(column_list)
        if c == "tco"
            extras_tensor[:,i] = crossover_time(car1, car2)
        else
            error("Unknown hand-constructed (extra) feature: $c")
        end
    end
    extras_tensor
end

function extra_features(car1::Array{T,3}, car2::Array{S,3}, column_list=String[]) where {T,S}
    extras_tensor = zeros(Float32, size(car1, 1), size(car1, 2), length(column_list))
    for (i, c) in enumerate(column_list)
        if c == "tco"
            extras_tensor[:,:,i] = crossover_time(car1, car2)
        else
            error("Unknown hand-constructed (extra) feature: $c")
        end
    end
    extras_tensor
end

visualize_predictions(data_dict) = visualize_predictions(data_dict["car1"],
                                                         data_dict["car2"],
                                                         get(data_dict, "extras_cols", String[]))
function visualize_predictions(car1, car2, extras_cols=String[])
    D = size(car1, 1)
    T = size(car1, 2)
    PH = 15
    Δt = 0.1
    @manipulate for data_id in 1:D,
                    t_predict in slider(1:T, value=10, label="t_predict"),
                    X in slider(0:256, value=0, label="Long. action"),
                    Y in slider(0:16, value=0, label="Lat. action"),
                    k in slider(0:100, value=50, label="# samples")
        tic()
        x0 = car1[data_id,t_predict+1:t_predict+3,1:2:end]
        y0 = car1[data_id,t_predict+1:t_predict+3,2:2:end]
        pfX, pfY = all_future_xy(xdd_choices, y_targets, car1[data_id,t_predict,:], tibvp, (x0, y0), 4)
        car1_future_input = zeros(1, 15, 6)
        if X == 0
            car1_future_input[1,:,1:2:end] = car1[data_id,t_predict + (1:PH),1:2:end]
        else
            car1_future_input[1,:,1:2:end] = pfX[X,2:end,:]
        end
        if Y == 0
            car1_future_input[1,:,2:2:end] = car1[data_id,t_predict + (1:PH),2:2:end]
        else
            car1_future_input[1,:,2:2:end] = pfY[Y,2:end,:]
        end
        TrafficWeavingPlanner.predict!(car1[data_id:data_id,:,:],
                                       car2[data_id:data_id,:,:],
                                       extra_features(car1[data_id:data_id,:,:], car2[data_id:data_id,:,:], extras_cols),
                                       [t_predict],
                                       car1_future_input,
                                       k)
        run_time = toq()
        pred_container = TrafficWeavingPlanner.pred_container
        x0, y0, xd0, yd0 = car2[data_id,t_predict,1:4]
        xdd = hcat(fill(car2[data_id,t_predict,5], size(pred_container.y,1)), pred_container.y[:,1,:,1])
        ydd = hcat(fill(car2[data_id,t_predict,6], size(pred_container.y,1)), pred_container.y[:,1,:,2])
        xd = trapz(xdd, Δt, xd0)
        yd = trapz(ydd, Δt, yd0)
        x = trapz(xd, Δt, x0)
        y = trapz(yd, Δt, y0)
        v = vel(xd, yd)
        a = acc(xd, xdd, yd, ydd, v)
        κ = curv(xd, xdd, yd, ydd, v)

        N = Int(sum(pred_container.z[1,1,:]))
        K = div(size(pred_container.z, 3), N)

        z_list = Vector{Int}(pred_container.z[:,1,:]*[i*K^j for j in 0:N-1 for i in 0:K-1]) + 1
        z_unique = 1:K^N

        color_list = distinguishable_colors(3+length(z_unique), [RGB(1,0,1), RGB(1,0,0), RGB(0,0,1)])[4:end]
        color_dict = Dict(z => c for (z,c) in zip(z_unique, color_list))
        colors = reshape([color_dict[z] for z in z_list], (1,length(z_list)))

        T_history = 1:t_predict
        T_future = t_predict + (0:PH)
        car1_history = car1[data_id,T_history,:]
        car2_history = car2[data_id,T_history,:]
        car1_future = [car1[data_id,t_predict,:]'; car1_future_input[1,:,:]]
        car2_future = car2[data_id,T_future,:]

        transparency = 0.7

        xy_left = plot([car1_history[:,1], car1_future[:,1], car2_history[:,1], car2_future[:,1]],
                       [car1_history[:,2], car1_future[:,2], car2_history[:,2], car2_future[:,2]],
                       linewidth=2,
                       color=[:blue :blue :red :red],
                       linestyles=[:solid :dash :solid :dash],
                       alpha=[1 1 1 ((X == 0 && Y == 0) ? 1 : 0)],
                       xlims=(-150,5), ylims=(-8,0),
                       title="position")
        plot!(xy_left, x', y', linewidth=.75, color=colors, alpha=transparency, legend=false)
        xy_right = plot(x', y', linewidth=.75, color=colors, alpha=transparency, legend=false, title="computation time = $(run_time)s")

        car1_history_v = vel(car1_history[:,3], car1_history[:,4])
        car2_history_v = vel(car2_history[:,3], car2_history[:,4])
        car1_future_v = vel(car1_future[:,3], car1_future[:,4])
        car2_future_v = vel(car2_future[:,3], car2_future[:,4])
        v_left = plot([Δt*T_history, Δt*T_future, Δt*T_history, Δt*T_future],
                      [car1_history_v, car1_future_v, car2_history_v, car2_future_v],
                      linewidth=2,
                      color=[:blue :blue :red :red],
                      linestyles=[:solid :dash :solid :dash],
                      alpha=[1 1 1 ((X == 0 && Y == 0) ? 1 : 0)],
                      title="longitudinal velocity")
        plot!(v_left, Δt*T_future, v', linewidth=.75, color=colors, alpha=transparency, legend=false)
        v_right = plot(Δt*T_future, v', linewidth=.75, color=colors, alpha=transparency, legend=false)

        car1_history_a = acc(car1_history[:,3], car1_history[:,5], car1_history[:,4], car1_history[:,6])
        car2_history_a = acc(car2_history[:,3], car2_history[:,5], car2_history[:,4], car2_history[:,6])
        car1_future_a = acc(car1_future[:,3], car1_future[:,5], car1_future[:,4], car1_future[:,6])
        car2_future_a = acc(car2_future[:,3], car2_future[:,5], car2_future[:,4], car2_future[:,6])
        a_left = plot([Δt*T_history, Δt*T_future, Δt*T_history, Δt*T_future],
                      [car1_history_a, car1_future_a, car2_history_a, car2_future_a],
                      linewidth=2,
                      color=[:blue :blue :red :red],
                      linestyles=[:solid :dash :solid :dash],
                      alpha=[1 1 1 ((X == 0 && Y == 0) ? 1 : 0)], 
                      xlims=(0, Δt*T_future[end]), ylims=(-6, 5),
                      title="longitudinal acceleration")
        plot!(a_left, Δt*T_future, a', linewidth=.75, color=colors, alpha=transparency, legend=false)
        a_right = plot(Δt*T_future, a', linewidth=.75, color=colors, alpha=transparency, legend=false)

        plot(xy_left, xy_right, v_left, v_right, a_left, a_right,
             layout=@layout([a b; c d; e f]), size=(800,900))
    end
end

function static_trial(data_dict, idx, xdd_choices, y_targets, tibvp, T=30; start_wait=8, extras_cols = String[])
    car1 = zeros(Float32, 1, T+15, 6)
    car1_curr_state = data_dict["car1"][idx,1,:]
    car1[1,1,:] = car1_curr_state

    if car1_curr_state[2] > -4
        l0, lf = 1, 2
    else
        l0, lf = 2, 1
    end

    pfX, pfY = all_future_xy(xdd_choices, y_targets, car1_curr_state, tibvp, [(1,l0) for i in 1:start_wait], 0)
    car1[1,1:start_wait+1,1:2:end] = pfX
    car1[1,1:start_wait+1,2:2:end] = pfY
    t = start_wait + 1 - 3
    car1_curr_state = car1[1,t,:]
    a0 = [(1,l0) for i in 1:3]
    while t < T
        pfX, pfY = all_future_xy(xdd_choices, y_targets, car1_curr_state, tibvp, a0, 4)
        car1_in = car1
        car2_in = data_dict["car2"][idx:idx,1:T+15,:]
        extras_in = extra_features(car1_in, car2_in, extras_cols)
        TrafficWeavingPlanner.predict!(car1_in, car2_in, extras_in, [t], pfX[:,2:end,:], pfY[:,2:end,:], 16)
#         @show norm(car1[1,t+1:t+3,:] - TrafficWeavingPlanner.pred_container.y[1:3,:])
#         @show pfX[1,:,3]
#         @show car1[1,t:t+6,idshow]
#         @show TrafficWeavingPlanner.pred_container.y[1:6,idshow]
        car1[1,t+1:t+15,:] = TrafficWeavingPlanner.pred_container.y
        t = t + 3
        car1_curr_state = car1[1,t,:]
        a0 = (TrafficWeavingPlanner.pred_container.y[4:6,1:2:end], TrafficWeavingPlanner.pred_container.y[4:6,2:2:end])
    end
    car1
end

function dynamic_trial(xdd_choices, y_targets, tibvp, initial_state=Float32[-136,-6,30,0,0,0]; T=100, start_wait=8, dt=Float32(.1), extras_cols = String[])
    hz = 100
    rate = TrafficWeavingPlanner.Rate(hz)
    TrafficWeavingPlanner.clear_human_states()
    TrafficWeavingPlanner.reset_simulator()
    for timeout in 1:hz
        length(TrafficWeavingPlanner.human_states.t) == 1 && break
        TrafficWeavingPlanner.rossleep(rate)
    end
    @assert length(TrafficWeavingPlanner.human_states.t) == 1
    t_offset = TrafficWeavingPlanner.human_states.t[1]

    car1 = zeros(Float32, 1, T, 6)
    car2 = zeros(Float32, 1, T, 6)
    car1_curr_state = initial_state
    car1[1,1,:] = car1_curr_state

    if car1_curr_state[2] > -4
        l0, lf = 1, 2
    else
        l0, lf = 2, 1
    end

    pfX, pfY = all_future_xy(xdd_choices, y_targets, car1_curr_state, tibvp, [(1,l0) for i in 1:start_wait], 0)
    car1[1,1:start_wait+1,1:2:end] = pfX
    car1[1,1:start_wait+1,2:2:end] = pfY
    t = start_wait + 1 - 3
    car1_curr_state = car1[1,t,:]
    a0 = [(1,l0) for i in 1:3]
    TrafficWeavingPlanner.publish_robot_path_plan(dt*(1:t+2), car1[1,2:t+3,1], car1[1,2:t+3,2])
    while (car1_curr_state[1] < 3 || TrafficWeavingPlanner.human_states.X[end][1] < 3) && t < T-15
        pfX, pfY = all_future_xy(xdd_choices, y_targets, car1_curr_state, tibvp, a0, 4)
        for timeout in 1:5*hz
            TrafficWeavingPlanner.human_states.t[end] > t_offset + dt*(t - 1) && break
            TrafficWeavingPlanner.rossleep(rate)
        end
        println((TrafficWeavingPlanner.human_states.t[end], t_offset + dt*(t - 1),
                 TrafficWeavingPlanner.human_states.t[end] - t_offset, dt*(t - 1)))
        @assert TrafficWeavingPlanner.human_states.t[end] > t_offset + dt*(t - 1)
        car2_interp = interpolate((TrafficWeavingPlanner.human_states.t,), TrafficWeavingPlanner.human_states.X, Gridded(Linear()))
        car2_in = [car2_interp[make_a_pr] for make_a_pr in t_offset+dt*(0:t-1)]
        @assert eltype(car2_in) == SVector{6,Float32}
        car2_in = reshape(reinterpret(Float32, car2_in, (6, length(car2_in)))', (1, length(car2_in), 6))
        car2[1,1:size(car2_in,2),:] = car2_in
        extras = extra_features(car1, car2, extras_cols)
        TrafficWeavingPlanner.predict!(car1, car2, extras, [t], pfX[:,2:end,:], pfY[:,2:end,:], 16)
        car1[1,t+1:t+15,:] = TrafficWeavingPlanner.pred_container.y
        t = t + 3
        car1_curr_state = car1[1,t,:]
        a0 = (TrafficWeavingPlanner.pred_container.y[4:6,1:2:end], TrafficWeavingPlanner.pred_container.y[4:6,2:2:end])
#         TrafficWeavingPlanner.publish_robot_path_plan(dt*(1:t+2), car1[1,2:t+3,1], car1[1,2:t+3,2])
#         TrafficWeavingPlanner.publish_robot_path_plan(dt*(1:t+5), car1[1,2:t+6,1], car1[1,2:t+6,2])
        TrafficWeavingPlanner.publish_robot_path_plan(dt*(1:t+8), car1[1,2:t+9,1], car1[1,2:t+9,2])
    end
    car1[:,2:t-3,:], car2[:,2:t-3,:]
end


### OLD SEMI-TRASH
# function random_coherent_actions(N, nx, ny, ply, a1_marginal, a2_marginal, starting_action=(1,1), stay_prob=.5)
# #     rand([(x,y) for x in 1:nx for y in 1:ny], N, ply)
# #     actions = Array{Tuple{Int,Int}}(N, ply)
#     actions = fill(starting_action, N, ply)
#     for c in 2:ply
#         for r in 1:N
#             if rand() < stay_prob
#                 actions[r,c] = actions[r,c-1]
#             else
#                 actions[r,c] = (rand(a1_marginal), rand(a2_marginal))
#             end
#         end
#     end
#     sortrows(actions)
# end

# random_actions(N, nx, ny, ply) = sortrows(rand([(x,y) for x in 1:nx for y in 1:ny], N, ply))

# @views function random_future(N, xdd_choices, y_targets, state0, tibvp,
#                               ply=5,
#                               actions=random_actions(N, length(xdd_choices), length(y_targets), ply),
#                               t_hold=3,
#                               dt=Float32(0.1))
#     x, y, ẋ, ẏ, ẍ, ÿ = 1:6

#     car_future = zeros(Float32, N, 1 + ply*t_hold, 6)
#     car_future[:,1,:] .= state0'
#     # xdd, xd, x
#     xdd_by_ply = [xdd_choices[i] for (i,j) in actions]
#     for i in 1:t_hold
#         car_future[:,1+i:t_hold:end,ẍ] = xdd_by_ply
#     end
#     cumsum!(car_future[:,2:end,ẋ], car_future[:,1:end-1,ẍ]*dt, 2)
#     car_future[:,2:end,ẋ] .+= car_future[:,1,ẋ]
#     cumsum!(car_future[:,2:end,x], (car_future[:,1:end-1,ẋ] .+ car_future[:,2:end,ẋ])*dt./2, 2)
#     car_future[:,2:end,x] .+= car_future[:,1,x]

#     # y_target, ydd, yd, y
# #     if state0[y] == y_final
# #         y_target = fill(y_final, N, ply)
# #     else
# # #         y_target = fill(y_start, N, ply)
# # #         for i in 1:N
# # #             y_target[i,rand(1:ply+1):end] = y_final
# # #         end
# #         y_target = rand([y_start, y_final], N, ply)
# #     end
#     y_target = [y_targets[j] for (i,j) in actions]
#     for i in 1:N
#         for p in 1:ply
#             curr_step = 1 + (p-1)*t_hold
#             q0 = SVector(car_future[i,curr_step,y],
#                          car_future[i,curr_step,ẏ],
#                          car_future[i,curr_step,ÿ])
#             qf = SVector(y_target[i,p], 0f0, 0f0)
#             c, u = tibvp(q0, qf, 100f0)
#             for t in 1:t_hold
#                 if dt*t > duration(u)
#                     q = qf
#                 else
#                     q = u.x(q0, u.y, u.t, t*dt)
#                 end
#                 car_future[i,curr_step+t,y] = q[1]
#                 car_future[i,curr_step+t,ẏ] = q[2]
#                 car_future[i,curr_step+t,ÿ] = q[3]
#             end
#         end
#     end
#     actions, car_future
# end

# function tree_reduce_max(actions, rewards)
#     if size(actions, 2) == 1
#         return 1, mean(rewards)
#     else
#         Nr = size(actions, 1)
#         split_points = [1]
#         prev_action = actions[1,1]
#         for r in 1:Nr
#             if actions[r,1] != prev_action
#                 push!(split_points, r)
#                 prev_action = actions[r,1]
#             end
#         end
#         push!(split_points, Nr+1)
#         return findmax(tree_reduce_max(view(actions, split_points[i]:split_points[i+1]-1, 2:size(actions, 2)), 
#                                        view(rewards, split_points[i]:split_points[i+1]-1))
#                        for i in 1:length(split_points)-1)
#     end
# end