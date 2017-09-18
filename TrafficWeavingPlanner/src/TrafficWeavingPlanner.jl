module TrafficWeavingPlanner

using RobotOS
using StaticArrays

@rosimport std_msgs.msg: Float32MultiArray, Float64MultiArray,
                         Int32MultiArray, Int64MultiArray,
                         MultiArrayLayout, MultiArrayDimension
@rosimport traffic_weaving_prediction.msg: prediction_input, prediction_output
@rosimport vtd_interface.msg: planner_result, rdb_object_state, scenario_start
rostypegen()

import std_msgs.msg: Float32MultiArray, Float64MultiArray,
                     Int32MultiArray, Int64MultiArray,
                     MultiArrayLayout, MultiArrayDimension
import traffic_weaving_prediction.msg: prediction_input, prediction_output
import vtd_interface.msg: planner_result, rdb_object_state, scenario_start

reversedims(A::Array) = permutedims(A, ndims(A):-1:1) # @generated probably the "right" way

for MAType in (Float32MultiArray, Float64MultiArray, Int32MultiArray, Int64MultiArray)
    @eval function Base.convert(::Type{$MAType}, A::Array{<:Real})
        Arev = reversedims(A)
        msg = $MAType()
        msg.layout.dim = [MultiArrayDimension("",d,s) for (d,s) in zip(size(A), reverse(cumprod([size(Arev)...])))]
        msg.data = Arev[:]
        msg
    end
    
    @eval function Base.convert(::Type{Array{T}}, msg::$MAType) where {T}
        if isempty(msg.data)
            T[]
        else
            reversedims(reshape(msg.data, (Int(d.size) for d in msg.layout.dim[end:-1:1])...))
        end
    end
end

mutable struct PredictionOutput
    stale::Bool
    y
    z
    r
end

function prediction_callback(msg::prediction_output, pred_container::PredictionOutput)
    pred_container.stale = false
    pred_container.y = Array{Float32}(msg.y)
    pred_container.z = Array{Float32}(msg.z)
    pred_container.r = Array{Float32}(msg.r)
end

init_node("julia_laneswap", anonymous=true)

const pred_container = PredictionOutput(true,nothing,nothing,nothing)
const publisher = Publisher{prediction_input}("prediction_input", queue_size=10)
const subscriber = Subscriber{prediction_output}("prediction_output", prediction_callback, (pred_container,), queue_size=10)

# rossleep(Duration(5))

function predict!(car1, car2, extras, traj_lengths, car1_future, sample_ct,
                  output::PredictionOutput=pred_container, timeout=1000)
    # println("predict! function running")
    output.stale = true
    publish(publisher, prediction_input(Float32MultiArray(car1),
                                        Float32MultiArray(car2),
                                        Float32MultiArray(extras),
                                        Int32MultiArray(traj_lengths),
                                        Float32MultiArray(car1_future),
                                        Float32MultiArray(Float32[]),
                                        Float32MultiArray(Float32[]),
                                        Int32(sample_ct)))
    # println("published data")
    for i in 1:timeout
        rossleep(Duration(0.001))
        output.stale || break
    end
    output
end

function predict!(car1, car2, extras, traj_lengths, car1_future_x, car1_future_y, sample_ct,
                  output::PredictionOutput=pred_container, timeout=1000)
    # println("predict! function running")
    output.stale = true
    publish(publisher, prediction_input(Float32MultiArray(car1),
                                        Float32MultiArray(car2),
                                        Float32MultiArray(extras),
                                        Int32MultiArray(traj_lengths),
                                        Float32MultiArray(Float32[]),
                                        Float32MultiArray(car1_future_x),
                                        Float32MultiArray(car1_future_y),
                                        Int32(sample_ct)))
    # println("published data")
    for i in 1:timeout
        rossleep(Duration(0.001))
        output.stale || break
    end
    output
end

####

mutable struct HumanCarStates
    t::Vector{Float32}
    X::Vector{SVector{6,Float32}}
end

function human_state_callback(msg::rdb_object_state, human_states::HumanCarStates)
    if isempty(human_states.t) || msg.simTime > human_states.t[end] + 1e-4
        push!(human_states.t, Float32(msg.simTime))
        push!(human_states.X, SVector{6,Float32}(msg.base.pos.x,
                                                 msg.base.pos.y,
                                                 msg.ext.speed.x,
                                                 msg.ext.speed.y,
                                                 msg.ext.accel.x,
                                                 msg.ext.accel.y))
    end
end

const human_states = HumanCarStates(Float32[], SVector{6,Float32}[])
const human_state_sub = Subscriber{rdb_object_state}("/object_state/1", human_state_callback, (human_states,), queue_size=10)
const planner_result_pub = Publisher{planner_result}("/planner_path", queue_size=10)
const scenario_setup_pub = Publisher{scenario_start}("/scenario_setup", queue_size=10)

function clear_human_states()
    human_states.t = Float32[]
    human_states.X = SVector{6,Float32}[]
    nothing
end

reset_simulator() = publish(planner_result_pub, planner_result(Float32[0], Float32[0], Float32[0]))
function setup_simulator(;xh=-136.0, yh=-1.83, vh=30.0, xr=-136.0, yr=-6.09, vr=30.0)
    publish(scenario_setup_pub, scenario_start(xh, yh, vh, xr, yr, vr))
end
publish_robot_path_plan(t, x, y) = publish(planner_result_pub, planner_result(t, x, y))

@spawn spin()

end # module
