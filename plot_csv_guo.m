% Guo Yu
% Plot results from csv file
% Please change the base's reference frame in FrankaEmikaPandaRobot.m!
% (try to combine several loops together
addpath(genpath('..\..\..\dqrobotics-toolbox-matlab'));
% position_guid = readmatrix('joint_position_exam_force_traj.csv'); 
% position_real = readmatrix('joint_position_real_joint_limit.csv'); 
% velocity_guid = readmatrix('joint_velocity_exam_force.csv'); 
% velocity_real = readmatrix('joint_velocity_real_joint_limit.csv'); 
position_guid = readmatrix('/home/guo/mani_qp_controller_vrep/mani_qp_controller_vrep/joint_velocity_commands/data/csv/joint_position_exam_force.csv'); 
position_real = readmatrix('/home/guo/mani_qp_controller_vrep/mani_qp_controller_vrep/joint_velocity_commands/data/csv/q_traj.csv'); 
velocity_guid = readmatrix('/home/guo/mani_qp_controller_vrep/mani_qp_controller_vrep/joint_velocity_commands/data/csv/joint_velocity_exam_force.csv'); 
velocity_real = readmatrix('/home/guo/mani_qp_controller_vrep/mani_qp_controller_vrep/joint_velocity_commands/data/csv/dq_traj.csv'); 
% xtrans_sigma = readmatrix('/home/gari/mani_check/src/mani_qp_controller/data/promp/xtrans_sigma_traj_3.csv');
% xtrans_mean = readmatrix('/home/gari/mani_check/src/mani_qp_controller/data/promp/xtrans_mean_traj.csv');
% xtrans_mean = readmatrix('/home/gari/mani_tracking_test/src/mani_qp_controller/data/csv/wrist_translation.csv');

xtrans_pose = readmatrix('./data/experiment/wrist_pose_shoulder_Drill_1.csv');
% remove time stamp and other useless columns
position_guid = position_guid(:,5:end);
position_real = position_real(:,1:end);
velocity_guid = velocity_guid(:,5:end);
velocity_real = velocity_real(:,1:end);
% xtrans_sigma = xtrans_sigma(:,1:end);
% [sigma_rows, sigma_cols] = size(xtrans_sigma);
% [xtrans_mean_rows, xtrans_mean_cols] = size(xtrans_mean);
% xtrans_mean = xtrans_mean(:,1:end);
[num_rows, num_columns] = size(position_guid);
[num_rows_real, num_columns_real] = size(position_real);
[num_rows_v, num_columns_v] = size(velocity_guid);
[num_rows_v_real, num_columns_v_real] = size(velocity_real);
[xtrans_pose_rows, xtrans_pose_cols] = size(xtrans_pose);
dt = 0.01;
linewidth_g=1;
linewidth_r=1;

% % Transpose DQ (8) to (6)
% xtrans_pose_ = xtrans_pose();

close all;
% Colors
clrmap = [  0.9970 0.6865 0.4692;
            0.1749 0.0670 0.3751;
            0.2 0.8 0.2];

% Plot joint position %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure (1)
hold on;
% guidance
plot([1:num_rows].*dt, position_guid(:,1), '--','color','r','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,2), '--','color','g','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,3), '--','color','b','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,4), '--','color','c','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,5), '--','color','m','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,6), '--','color','#D95319','Linewidth',linewidth_g);
plot([1:num_rows].*dt, position_guid(:,7), '--','color','k','Linewidth',linewidth_g);
% real traj
plot([1:num_rows_real].*dt, position_real(:,1), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,2), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,3), '-','color','b','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,4), '-','color','c','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,5), '-','color','m','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,6), '-','color','#D95319','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, position_real(:,7), '-','color','k','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
legend('q_1','q_2','q_3','q_4','q_5','q_6','q_7');
xlabel('Time t(s)','fontsize',22,'Interpreter','latex');
ylabel(['Joint Position q(rad)'],'fontsize',22,'Interpreter','latex');
% title(['Plot of joint position during guidance'],'fontsize',28);
grid on;

% Plot joint velocity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
velocity_guid_ = diff(position_guid,1,1)/dt;
velocity_real_ = diff(position_real,1,1)/dt;

figure(2)
hold on;
% guidance
% plot([1:num_rows_v].*dt, velocity_guid(:,1), '--','color','r','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,2), '--','color','g','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,3), '--','color','b','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,4), '--','color','c','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,5), '--','color','m','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,6), '--','color','#D95319','Linewidth',linewidth_g);
% plot([1:num_rows_v].*dt, velocity_guid(:,7), '--','color','k','Linewidth',linewidth_g);
% real traj
plot([1:num_rows_v_real].*dt, velocity_real(:,1), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,2), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,3), '-','color','b','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,4), '-','color','c','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,5), '-','color','m','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,6), '-','color','#D95319','Linewidth',linewidth_r);
plot([1:num_rows_v_real].*dt, velocity_real(:,7), '-','color','k','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
legend('dq_1','dq_2','dq_3','dq_4','dq_5','dq_6','dq_7');
xlabel('Time t(s)','fontsize',22,'Interpreter','latex');
ylabel(['Joint velocity dq(rad/s)'],'fontsize',22,'Interpreter','latex');
% title(['Plot of joint velocity during guidance'],'fontsize',28);
grid on;

% Plot joint acceleration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acc_guid_ = diff(velocity_guid_,1,1)/dt;
% acc_real_ = diff(velocity_real_,1,1)/dt;
acc_guid = diff(velocity_guid,1,1)/dt;
acc_real = diff(velocity_real,1,1)/dt;

figure(3)
hold on;
% guidance
% plot([1:num_rows_v-1].*dt, acc_guid(:,1), '--','color','r','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,2), '--','color','g','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,3), '--','color','b','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,4), '--','color','c','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,5), '--','color','m','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,6), '--','color','#D95319','Linewidth',linewidth_g);
% plot([1:num_rows_v-1].*dt, acc_guid(:,7), '--','color','k','Linewidth',linewidth_g);
% % real traj
plot([1:num_rows_v_real-1].*dt, acc_real(:,1), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,2), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,3), '-','color','b','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,4), '-','color','c','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,5), '-','color','m','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,6), '-','color','#D95319','Linewidth',linewidth_r);
plot([1:num_rows_v_real-1].*dt, acc_real(:,7), '-','color','k','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
legend('ddq_1','ddq_2','ddq_3','ddq_4','ddq_5','ddq_6','ddq_7');
xlabel('Time t(s)','fontsize',22,'Interpreter','latex');
ylabel('Joint acceleration ddq($rad/s^{2}$)','fontsize',22,'Interpreter','latex');
% title(['Plot of joint acceleration during guidance']);
grid on;

% Plot cartesian position %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robot = FrankaEmikaPandaRobot.kinematics();
xt_traj_guid = zeros(3,num_rows);
xt_traj_real = zeros(3,num_rows_real);
xt_traj_guid_rot = zeros(4,num_rows);
xt_traj_real_rot = zeros(4,num_rows_real);
for row = 1:num_rows
    % forward kinematic (guid)
    xt = robot.fkm(position_guid(row,:));
    xt_tran = vec3(translation(xt));
    xt_rot = vec4(rotation(xt));
    xt_traj_guid(:,row) = xt_tran;
    xt_traj_guid_rot(:,row) = xt_rot;
end
for row_real = 1:num_rows_real
    % forward kinematic (real)
    xt_real = robot.fkm(position_real(row_real,:));
    xt_tran_real = vec3(translation(xt_real));
    xt_rot_real = vec4(rotation(xt_real));
    xt_traj_real(:,row_real) = xt_tran_real;
    xt_traj_real_rot(:,row_real) = xt_rot_real;
end
% wrist pose (experiemnt cartesian pose)
for row_exp = 1:xtrans_pose_rows
    % forward kinematic
    xt_exp = DQ(xtrans_pose(row_exp,:));
    xt_tran = vec3(translation(xt_exp));
    xt_rot = vec4(rotation(xt_exp));
    xt_traj_exp_tran(row_exp,:) = xt_tran;
    xt_traj_exp_rot(row_exp,:) = xt_rot;
end

figure(4)
hold on;
% guidance
plot([1:num_rows].*dt, xt_traj_guid(1,:), '--','color','r','Linewidth',linewidth_g);
plot([1:num_rows].*dt, xt_traj_guid(2,:), '--','color','g','Linewidth',linewidth_g);
plot([1:num_rows].*dt, xt_traj_guid(3,:), '--','color','b','Linewidth',linewidth_g);
% % real traj
plot([1:num_rows_real].*dt, xt_traj_real(1,:), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, xt_traj_real(2,:), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, xt_traj_real(3,:), '-','color','b','Linewidth',linewidth_r);
% plot([1:num_rows_real].*dt, xt_traj_real(1,:)  + dot_x(1,:), '-','color','r','Linewidth',linewidth_r);
% plot([1:num_rows_real].*dt, xt_traj_real(2,:) + dot_x(2,:), '-','color','g','Linewidth',linewidth_r);
% plot([1:num_rows_real].*dt, xt_traj_real(3,:)  + dot_x(3,:), '-','color','b','Linewidth',linewidth_r);
% % guidance
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_tran(:,1), '--','color','r','Linewidth',linewidth_r);
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_tran(:,2), '--','color','g','Linewidth',linewidth_r);
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_tran(:,3), '--','color','b','Linewidth',linewidth_r);
% 
% % real traj
% plot([1:num_rows_real].*dt, xt_traj_real(1,:), '-','color','r','Linewidth',linewidth_r);
% plot([1:num_rows_real].*dt, xt_traj_real(2,:), '-','color','g','Linewidth',linewidth_r);
% plot([1:num_rows_real].*dt, xt_traj_real(3,:), '-','color','b','Linewidth',linewidth_r);

% t = [1:sigma_rows].*dt;
% x_lb = xtrans_mean(:,1)  - xtrans_sigma(:,1);
% x_ub = xtrans_mean(:,1) + xtrans_sigma(:,1);
% y_lb = xtrans_mean(:,2) - xtrans_sigma(:,2);
% y_ub = xtrans_mean(:,2) + xtrans_sigma(:,2);
% z_lb = xtrans_mean(:,3) - xtrans_sigma(:,3);
% z_ub = xtrans_mean(:,3) + xtrans_sigma(:,3);

% plot(t, x_lb, 'r', 'LineStyle','none');
% plot(t, x_ub, 'r', 'LineStyle','none');
% t2 = [t, fliplr(t)];
% inBetween_x = [x_lb', flip(x_ub)'];
% inBetween_y = [y_lb', flip(y_ub)'];
% inBetween_z = [z_lb', flip(z_ub)'];
% fill(t2, inBetween_x, 'g','FaceColor','r','FaceAlpha',.3,'EdgeAlpha',.3);
% fill(t2, inBetween_y, 'g','FaceColor','g','FaceAlpha',.3,'EdgeAlpha',.3);
% fill(t2, inBetween_z, 'g','FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3);

% y2 = 4 + cos(x).*exp(0.1*x);
% y_lb = ;
% y_ub = xt_traj_real
% hold on
% area(x,y2,'FaceColor','r','FaceAlpha',.3,'EdgeAlpha',.3)

set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
legend('x','y','z');
xlabel('Time t(s)','fontsize',22,'Interpreter','latex');
ylabel(['Cartesian Position (m)'],'fontsize',22,'Interpreter','latex');
% title(['Cartesian position during guidance'],'fontsize',28);
grid on;


% Plot cartesian orientation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(5)
hold on;
xt_traj_real_rot(1,:);
% guidance
plot([1:num_rows].*dt, xt_traj_guid_rot(1,:), '--','color','r','Linewidth',linewidth_g);
plot([1:num_rows].*dt, xt_traj_guid_rot(2,:), '--','color','g','Linewidth',linewidth_g);
plot([1:num_rows].*dt, xt_traj_guid_rot(3,:), '--','color','b','Linewidth',linewidth_g);
plot([1:num_rows].*dt, xt_traj_guid_rot(4,:), '--','color','c','Linewidth',linewidth_g);
% real traj
plot([1:num_rows_real].*dt, xt_traj_real_rot(1,:), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, xt_traj_real_rot(2,:), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, xt_traj_real_rot(3,:), '-','color','b','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, xt_traj_real_rot(4,:), '-','color','c','Linewidth',linewidth_r);
% experiment data
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_rot(:,1), '-','color','r','Linewidth',linewidth_r);
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_rot(:,2), '-','color','g','Linewidth',linewidth_r);
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_rot(:,3), '-','color','b','Linewidth',linewidth_r);
% plot([1:xtrans_pose_rows].*dt, xt_traj_exp_rot(:,4), '-','color','c','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',28,'Interpreter','latex');
% ylabel('$q$','fontsize',28,'Interpreter','latex');
legend('real','i','j','k');
xlabel('Time t(s)','fontsize',22,'Interpreter','latex');
ylabel(['Cartesian Orientation'],'fontsize',22,'Interpreter','latex');
% title(['Cartesian orientation during guidance'],'fontsize',28);
grid on;

% Plot distance between current and desired ME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = zeros(num_rows_real,1);
ev_real = zeros(6,num_rows_real);
ev_guid = zeros(6,num_rows_real);
J_xy = zeros(2,7);
J_xz = zeros(2,7);
J_yz = zeros(2,7);
J_xy_d = zeros(2,7);
J_xz_d = zeros(2,7);
J_yz_d = zeros(2,7);
gmm_c =[];
for row_real = 1:num_rows_real
    % ME: real traj
    Jt_geom = geomJ(robot,position_real(row_real,:));
    Me_ct = Jt_geom * Jt_geom';
    % singular value of Jacobian
    ev_real(:,row_real) = svd(Jt_geom);

    % ME: from guidance
    Jt_geom_d = geomJ(robot,position_guid(row_real,:));
    Me_d = Jt_geom_d * Jt_geom_d'; 
    % singular value of Jacobian
    ev_guid(:,row_real) = svd(Jt_geom_d);
    % distance between real and desired
    d(row_real,1) = norm(logm(Me_d^-.5*Me_ct*Me_d^-.5),'fro');
end

% Plot Manipulability Ellipsoid %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cartesian cartesian velocity (T) 
Jt_geom_t = zeros(3,7);
Me_proj = zeros(3,3);

xt_xy_traj = xt_traj_real(1:2,:);
xt_xz_traj = zeros(2,num_rows_real);
xt_xz_traj(1,:) = xt_traj_real(1,:);
xt_xz_traj(2,:) = xt_traj_real(3,:);
xt_yz_traj = xt_traj_real(2:3,:);

xt_xy = zeros(2,1);
xt_xz = zeros(2,1);
xt_yz = zeros(2,1);
xt_xy_guid = zeros(2,1);
xt_xz_guid = zeros(2,1);
xt_yz_guid = zeros(2,1);
dxt_xy = zeros(2,1);
dxt_xz = zeros(2,1);
dxt_yz = zeros(2,1);
counter = 0;

for row_real = 1:200:num_rows_real
    counter = counter + 1;
    % ME: real traj
    Jt_geom = geomJ(robot,position_real(row_real,:));
    J_xy = Jt_geom(4:5,:);
    J_xz(1,:) = Jt_geom(4,:);
    J_xz(2,:) = Jt_geom(6,:);
    J_yz = Jt_geom(5:6,:);
    Me_ct = Jt_geom * Jt_geom';
    Me_ct_xy = J_xy * J_xy';
    Me_ct_xz = J_xz * J_xz';
    Me_ct_yz = J_yz * J_yz';

    % ME: from guidance
    Jt_geom_d = geomJ(robot,position_guid(row_real,:));
    J_xy_d = Jt_geom_d(4:5,:);
    J_xz_d(1,:) = Jt_geom_d(4,:);
    J_xz_d(2,:) = Jt_geom_d(6,:);
    J_yz_d = Jt_geom_d(5:6,:);
    Me_d = Jt_geom_d * Jt_geom_d'; 
    Me_d_xy = J_xy_d * J_xy_d';
    Me_d_xz = J_xz_d * J_xz_d';
    Me_d_yz = J_yz_d * J_yz_d';

    % xt
    xt_xy = xt_traj_real(1:2,row_real);
    xt_xz(1,1) = xt_traj_real(1,row_real);
    xt_xz(2,1) = xt_traj_real(3,row_real);
    xt_yz = xt_traj_real(2:3,row_real);
    
    xt_xy_guid = xt_traj_guid(1:2,row_real);
    xt_xz_guid(1,1) = xt_traj_guid(1,row_real);
    xt_xz_guid(2,1) = xt_traj_guid(3,row_real);
    xt_yz_guid = xt_traj_guid(2:3,row_real);
    
    figure(6)
    hold on;
    colTmp = [1-0.1, 1-0.1, 1-0.1] - [.05,.05,.05] * counter;
    transpFactor = .05 + 0.02*counter;
    edgFactor = .5 + 0.03*counter;
    plotGMM(xt_xy, 1E-2*Me_d_xy, clrmap(3,:), transpFactor, '-.', 2, edgFactor); % Scaled matrix!
    plotGMM(xt_xy, 1E-2*Me_ct_xy, colTmp, .4, '-', 3, 1); % Scaled matrix!
    set(gca,'FontSize',14);
    % axis square;
    % axis equal;
    xlabel('$x$','fontsize',38,'Interpreter','latex');
    ylabel('$y$','fontsize',38,'Interpreter','latex');
    title(['Visualized actual and desired ME'],'FontSize',28,'Interpreter','latex');

    figure(7)
    hold on;
    plotGMM(xt_xz, 1E-2*Me_d_xz, clrmap(3,:), transpFactor, '-.', 2, edgFactor); % Scaled matrix!
    plotGMM(xt_xz, 1E-2*Me_ct_xz, colTmp, .4, '-', 3, 1); % Scaled matrix!
    set(gca,'FontSize',14);
    % axis square;
    % axis equal;
    xlabel('$x$','fontsize',38,'Interpreter','latex');
    ylabel('$z$','fontsize',38,'Interpreter','latex');
    title(['Visualized actual and desired ME'],'FontSize',28,'Interpreter','latex');

    figure(8)
    hold on;
    plotGMM(xt_yz, 1E-2*Me_d_yz, clrmap(3,:), transpFactor, '-.', 2, edgFactor); % Scaled matrix!
    plotGMM(xt_yz, 1E-2*Me_ct_yz, colTmp, .4, '-', 3, 1); % Scaled matrix!
    set(gca,'FontSize',14);
    % axis square;
    % axis equal;
    xlabel('$y$','fontsize',38,'Interpreter','latex');
    ylabel('$z$','fontsize',38,'Interpreter','latex');
    title(['Visualized actual and desired ME'],'FontSize',28,'Interpreter','latex');
    % drawnow;

end

%% ME time sequence visualization
figure(6)
hold on;
n_color=100;
% p=plot(xt_xy_traj(1,:),xt_xy_traj(2,:),'-');
% cd = [uint8(jet(n_color)*255) uint8(ones(n_color,1))].'; 
% % drawnow 
% set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd) 
sizeMarker = linspace(0.1, 50, num_rows_real);  
colorMarker = linspace(0.9, 0.3, length(xt_xy_traj(1,:)));  % 颜色渐变
colTmp = zeros(num_rows_real,3);
colTmp(:,1)=colorMarker;
colTmp(:,2)=colorMarker;
colTmp(:,3)=colorMarker;
scatter(xt_xy_traj(1,:),xt_xy_traj(2,:), sizeMarker,colTmp, 'o', 'filled')

figure(7)
hold on;
% plot(xt_xz_traj(1,:),xt_xz_traj(2,:),'-');
scatter(xt_xz_traj(1,:),xt_xz_traj(2,:), sizeMarker,colTmp, 'o', 'filled')

figure(8)
hold on;
% plot(xt_yz_traj(1,:),xt_yz_traj(2,:),'-');
scatter(xt_yz_traj(1,:),xt_yz_traj(2,:), sizeMarker,colTmp, 'o', 'filled')

% Plot projected ME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for row_real = 1:num_rows_real
% % projected ME
%     dxt_xy = dxt_traj_real(1:2,row_real);
%     dxt_xz(1,1) = dxt_traj_real(1,row_real);
%     dxt_xz(2,1) = dxt_traj_real(3,row_real);
%     dxt_yz = dxt_traj_real(2:3,row_real);
%     Me_proj_xy = dxt_xy' * (J_xy*J_xy')*dxt_xy
%     Me_proj_xz = dxt_xz' * (J_xz*J_xz')*dxt_xz;
%     Me_proj_yz = dxt_yz' * (J_yz*J_yz')*dxt_yz;
% end
% cartesian velocity traj
dxt_traj_real = zeros(3,num_rows_real-1);
proj_me = zeros(1,num_rows_real-1);
for i = 2:num_rows_real
    dxt_traj_real(:,i-1) = (xt_traj_real(:,i) - xt_traj_real(:,i-1))/dt;
    Jt_geom = geomJ(robot,position_real(i,:));
    Jt_geom_t = Jt_geom(4:6,:);
    proj_me(1,i-1) = dxt_traj_real(:,i-1)' * (Jt_geom_t*(Jt_geom_t'))*dxt_traj_real(:,i-1);
end

figure(9)
hold on;
% proj. manipulability ellipsoid
plot([1:num_rows_real-1].*dt, proj_me(1,:), '-','color','r','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
legend('x','y','z');
xlabel('Time t(s)','FontSize',28);
ylabel(['Manipulability'],'FontSize',28);
title(['Plot of projected ME']);
grid on;


% % Desired and initial manipulability ellipsoids
% figure('position',[10 10 450 450],'color',[1 1 1]);
% hold on;
% plotGMM([0;0], 1E-2*Me_d_xy, clrmap(3,:), .5, '-.', 3, 1);
% plotGMM([0;0], 1E-2*Me_track(:,:,1), clrmap(1,:), .3, '--', 3, 1);
% xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
% xlim([-2 2]);ylim([-2 2]);
% % xlim([-1.2 1.2]);ylim([-1.2 1.2]);
% set(gca,'xtick',[],'ytick',[]);
% text(-.8,1,0,'Initial','FontSize',38,'Interpreter','latex')
% axis equal;
% 
% % Desired and final manipulability ellipsoids
% figure('position',[10 10 450 450],'color',[1 1 1]);
% hold on;
% plotGMM([0;0], 1E-2*Me_d, clrmap(3,:), .6, '-.', 3, 1);
% plotGMM([0;0], 1E-2*Me_ct, clrmap(2,:), .3, '-', 3, 1);
% xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
% xlim([-2 2]);ylim([-2 2]);
% % xlim([-1.2 1.2]);ylim([-1.2 1.2]);
% text(-.7,1,0,'Final','FontSize',38,'Interpreter','latex')
% set(gca,'xtick',[],'ytick',[])
% axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(10)
hold on;
% distance
plot([1:num_rows_real].*dt, d(:,1), '-','color','r','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
xlabel('Time t(s)','FontSize',22,'Interpreter','latex');
ylabel('Distance d','FontSize',22,'Interpreter','latex');
title(['Distance between actual and desired ME'],'FontSize',28,'Interpreter','latex');
grid on;

% Plot singular value of Jacobian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (check if min. sing value satisfied) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(11)
hold on;
% guidance
plot([1:num_rows_real].*dt, ev_guid(1,:), '--','color','r','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_guid(2,:), '--','color','g','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_guid(3,:), '--','color','b','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_guid(4,:), '--','color','c','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_guid(5,:), '--','color','m','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_guid(6,:), '--','color','#D95319','Linewidth',linewidth_r);
% real
plot([1:num_rows_real].*dt, ev_real(1,:), '-','color','r','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_real(2,:), '-','color','g','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_real(3,:), '-','color','b','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_real(4,:), '-','color','c','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_real(5,:), '-','color','m','Linewidth',linewidth_r);
plot([1:num_rows_real].*dt, ev_real(6,:), '-','color','#D95319','Linewidth',linewidth_r);
set(gca,'fontsize',14);
xlim([0 num_rows*dt])
% xlabel('$t$','fontsize',22,'Interpreter','latex');
% ylabel('$q$','fontsize',22,'Interpreter','latex');
legend('ev_1','ev_2','ev_3','ev_4','ev_5','ev_6');
xlabel('Time t(s)','FontSize',22,'Interpreter','latex');
ylabel(['Eigenvalue'],'FontSize',22,'Interpreter','latex');
title(['Eigenvalue of ME'],'FontSize',28,'Interpreter','latex');
grid on;


