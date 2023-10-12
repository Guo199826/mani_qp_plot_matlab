robot = FrankaEmikaPandaRobot.kinematics();


dq_jaco = readmatrix('human_data/human_Jacobian/geometrical_Jacobian_Drill_1.csv');
s = size(dq_jaco);
human_jaco_vol = zeros(s(1),1);

robot_pos = readmatrix("human_data/robot/q_position_mean_traj.csv");
s2 = size(robot_pos);
robot_jaco_vol = zeros(s2(1),1);

for i = 1:s
    i_jaco_col = dq_jaco(i,:)/100;
    i_jaco = reshape(i_jaco_col,8,8);
    human_mani = i_jaco * transpose(i_jaco);
    eigen_human = eig(human_mani);
    human_vol = prod(eigen_human, "all");
    human_jaco_vol(i,:) = human_vol;


    robot_jaco = kine.pose_jacobian(robot_pos(i, :));
    robot_mani = robot_jaco * transpose(robot_jaco);
    eigen_robot = eig(robot_mani);
    robot_vol = prod(eigen_robot, "all");
    robot_jaco_vol(i,:) = robot_vol;

    % geo_jaco = geomJ(robot,position_real(i,:));
    % geo_jaco_vol = z
end

plot(human_jaco_vol)
hold on
plot(robot_jaco_vol)
legend('human','robot')
title('human mani volume/100 & robot mani volume')



