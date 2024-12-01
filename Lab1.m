clc;
clear all;
close all;

disp('Haim Cohen 207070152')

%1.a
%% 

t=0.2:1/100:3; 
wm=6*pi;
x = 4./(wm*pi*t.^2).*(sin(wm*t./2)).^2.*(cos(wm/2*t)).*(sin(wm*t)) ;
figure;                             
plot(t,abs(x));
title('Question 1.a');
xlabel('t [s]','FontSize',10); ylabel('|x(t)| [V]','FontSize',10);
legend({'x(t)'});

%1.b
%% 

w=-17*pi:1/100:17*pi; 
triangle_minus3 = triangularPulse((w-3/2*wm)/(wm));
triangle_plus1 = -triangularPulse((w+wm/2)/(wm));
triangle_minus1 = triangularPulse((w-wm/2)/(wm));
triangle_plus3 = -triangularPulse((w+3/2*wm)/(wm));

figure;                             
plot(w,triangle_minus3, 'r');
hold on
plot(w,triangle_plus1, 'b');
hold on
plot(w,triangle_minus1, 'g');
hold on
plot(w,triangle_plus3, 'y');
title('Question 1.b');
xlabel('w [rad/sec]','FontSize',10); 
ylabel('|X(w)|','FontSize',10);
xline(15*pi, 'r--','$x = 15pi$','LineWidth', 1,'Interpreter', 'latex','LabelOrientation', 'horizontal');
grid on;
legend([{'Triangle[(w-3/2*wm)/wm)]'};{'Triangle[(w+wm/2)/wm)]'};{'Triangle[(w-wm/2)/wm)]'};{'Triangle[(w+3/2*wm)/wm)]'}]);

figure;
Xf = triangle_minus3 + triangle_plus1 + triangle_minus1 + triangle_plus3;
plot(w,abs(Xf), 'b');
title('|X(W)|');
xlabel('w [rad/sec]','FontSize',10); ylabel('|X(w)|','FontSize',10);
xline(15*pi, 'r--','$x = 15pi$','LineWidth', 1,'Interpreter', 'latex','LabelOrientation', 'horizontal');
grid on;
legend({'|X(w)|'});

%1.c
%%
figure;
nTs=0.2:1/15:3; 
xnTs = 4./(wm*pi*nTs.^2).*(sin(wm*nTs./2)).^2.*(cos(wm/2*nTs)).*(sin(wm*nTs)) ;
stairs(nTs,xnTs);
hold on;
plot(t,x, 'r');
grid;
legend([{'ZOH'};{'x'}]);
ylim([-0.5 1]);
title('x_Z_O_H(t) 1.c');
xlabel('t [sec]','FontSize',10);
ylabel('[V]','FontSize',10);

%1.d
%% 

figure;
ws = 5*wm;
withoutSinc =   ...
        triangularPulse(wm/2-ws,3/2*wm-ws,5/2*wm-ws,w) - ...
        triangularPulse(-5/2*wm-ws,-3/2*wm-ws,-wm/2-ws,w) + ...
        triangularPulse(-wm/2-ws,wm/2-ws,3/2*wm-ws,w) - ...
        triangularPulse(-3/2*wm-ws,-wm/2-ws,wm/2-ws,w) + ...  
        triangularPulse(wm/2,3/2*wm,5/2*wm,w) - ...
        triangularPulse(-5/2*wm,-3/2*wm,-wm/2,w) + ...
        triangularPulse(-wm/2,wm/2,3/2*wm,w) - ...
        triangularPulse(-3*wm/2,-wm/2,wm/2,w) + ...
        triangularPulse(wm/2+ws,3/2*wm+ws,5/2*wm+ws,w) - ...
        triangularPulse(-5/2*wm+ws,-3/2*wm+ws,-wm/2+ws,w) + ...
        triangularPulse(-wm/2+ws,wm/2+ws,3/2*wm+ws,w) - ...
        triangularPulse(-3/2*wm+ws,-wm/2+ws,wm/2+ws,w);

Xzoh = sinc(w./(30*pi)).*withoutSinc;
plot(w,abs(Xzoh), 'b');
grid;
title('|ZOH(w)| 1.d');
xlabel('w[rad/sec]','FontSize',10);
ylabel('|XZOH(w)|','FontSize',10);

%1.e
%%

t=0.2:1/100:3; 
w=-17*pi:1/100:17*pi; 
nTs=0.2:1/15:3; 
xOfnTs = 4./(wm*pi*nTs.^2).*(sin(wm*nTs./2)).^2.*(cos(wm/2*nTs)).*(sin(wm*nTs)) ;

triangle_minus3_n = (1/2i)*triangularPulse((w-3/2*wm)/(wm));
triangle_plus1_n = -(1/2i)*triangularPulse((w+wm/2)/(wm));
triangle_minus1_n = (1/2i)*triangularPulse((w-wm/2)/(wm));
triangle_plus3_n = -(1/2i)*triangularPulse((w+3/2*wm)/(wm));

Xf = triangle_minus3_n + triangle_plus1_n + triangle_minus1_n + triangle_plus3_n;

x_tild = zeros(length(t));
    for t_idx = 1:1:length(t)
       x_tild(t_idx) = 1/(2*pi)*trapz(w,Xf.*exp(1i*w*t(t_idx)));
    end  
figure; 
s = stairs(nTs,abs(xOfnTs));
s.Color = 'green';
hold on;
plot(t,abs(x), 'r');
hold on;
plot(t,abs(x_tild), '--k','LineWidth',2);

title('|xrec(t)| 1.e');
xlabel('t[sec]','FontSize',10); 
ylabel('[V]','FontSize',10);
legend([{'|xZOH(t)|'};{'|x(t)|'};{'|xrec(t)|'}]);
grid

%1.f
%%
%PartA
figure; 
wm=6*pi;
ws = 4*wm;
w=-17*pi:1/100:17*pi; 
nTs=0.2:1/15:3; 
xOfnTs = 4./(wm*pi*nTs.^2).*(sin(wm*nTs./2)).^2.*(cos(wm/2*nTs)).*(sin(wm*nTs)) ;
withoutSinc =   ...
        triangularPulse(wm/2-ws,3/2*wm-ws,5/2*wm-ws,w) - ...
        triangularPulse(-5/2*wm-ws,-3/2*wm-ws,-wm/2-ws,w) + ...
        triangularPulse(-wm/2-ws,wm/2-ws,3/2*wm-ws,w) - ...
        triangularPulse(-3/2*wm-ws,-wm/2-ws,wm/2-ws,w) + ...  
        triangularPulse(wm/2,3/2*wm,5/2*wm,w) - ...
        triangularPulse(-5/2*wm,-3/2*wm,-wm/2,w) + ...
        triangularPulse(-wm/2,wm/2,3/2*wm,w) - ...
        triangularPulse(-3*wm/2,-wm/2,wm/2,w) + ...
        triangularPulse(wm/2+ws,3/2*wm+ws,5/2*wm+ws,w) - ...
        triangularPulse(-5/2*wm+ws,-3/2*wm+ws,-wm/2+ws,w) + ...
        triangularPulse(-wm/2+ws,wm/2+ws,3/2*wm+ws,w) - ...
        triangularPulse(-3/2*wm+ws,-wm/2+ws,wm/2+ws,w);

x_tild = zeros(length(t));
    for t_idx = 1:1:length(t)
       x_tild(t_idx) = 1/(2*pi)*trapz(w,withoutSinc.*exp(1i*w*t(t_idx)));
    end  
plot(t,abs(x), 'r');
hold on;
plot(t,abs(x_tild), '--b','LineWidth',1);
hold on

title('|x(t)| and  |xrec(t)| with m_s of 4*w_m 1.f');
xlabel('t [sec]','FontSize',10); ylabel('[V]','FontSize',10);
legend([{'|x(t)|'};{'|xrec(t)|, w_s=4w_m'}]);
grid

%2.a
%%
wa = 5*pi;
wb = 2*pi;
T = 2;
w0 = 2*pi/T;
nsmpl = 11;
t = 0:1/100:T;
x_t = 5*cos(wa*t)-3*sin(wb*t);
t_s(1,:) = linspace(T/nsmpl,T,nsmpl);
x_s = zeros(size(t_s,1),nsmpl);
for i = 1:size(t_s,1)
   x_s(i,:) = 5*cos(wa*t_s(i,:))-3*sin(wb*t_s(i,:));
end
figure
plot(t,x_t,'b');
hold on;
stem(t_s(1,:),x_s(1,:),'r')
title('x(t) and x_s 2a')
xlabel('t[sec]')
ylabel('Amplitude[V]')
legend('x(t)','xs(t)')

%2.b + 2.c
%%
wb = 2*pi;
wa = 5*pi;
T = 2;
w0 = 2*pi/T;
n_sample = 11; 
t = 0:1/100:T;
x_t = 5*cos(wa*t)-3*sin(wb*t);
M = 5;
t_s(1,:) = linspace(T/n_sample,T,n_sample);
a = zeros(size(t_s,1),n_sample);
cond_num = zeros(size(t_s,1),1);

x_s(1,:) = 5*cos(wa*t_s(1,:))-3*sin(wb*t_s(1,:));
F = zeros(n_sample,M);
for n = 1:n_sample
   for m = 1:2*M+1
       F(n,m) = exp(1i*(t_s(1,n)+(0.01*rand(1)))*(-5+m-1)*w0);
   end
end
if n_sample == 2*M+1
   a(1,:) = inv(F)*x_s(1,:)';
elseif n_sample>2*M+1
   a(1,:) = inv(F'*F)*F'*x_s(1,:)';
end
cond_num(1) = cond(F);


t = 0:1/100:T;
x_res = zeros(size(t_s,1),length(t));
for k = 1:2*M+1
    x_res(1,:) = x_res(1,:) + a(1,k)*exp(1i*(k-M-1)*w0*t);
end  

figure
plot(t,x_t,'k')
hold on
plot(t,x_res(1,:),'--r', 'LineWidth',2)
legend('x(t)', 'x^~(t)'); title('restored x(t) 2.c')
xlabel('t[sec]','FontSize',10); ylabel('[V]','FontSize',10);

%2.d
%%
M = 5;
wb = 2*pi;
wa = 5*pi;
T = 2;
w0 = 2*pi/T;
nsmpl = 11;
t_s(2,:) = T*rand([1,nsmpl]);
x_s(2,:) = 5*cos(wa*t_s(2,:))-3*sin(wb*t_s(2,:));
x_res = zeros(size(t_s,1),length(t));
x_t = 5*cos(wa*t)-3*sin(wb*t);
F = zeros(nsmpl,M);
for n = 1:nsmpl
   for m = 1:2*M+1
       F(n,m) = exp(1i*(t_s(2,n))*(-5+m-1)*w0) ;  
   end
end
if nsmpl == 2*M+1
   a(2,:) = inv(F)*x_s(2,:)';
elseif nsmpl>2*M+1
   a(2,:) = inv(F'*F)*F'*x_s(2,:)';
end
cond_num(2) = cond(F);

for k = 1:2*M+1
    x_res(2,:) = x_res(2,:) + a(2,k)*exp(1i*(k-M-1)*w0*t);
end  
cond_num(2)

figure
plot(t,x_t,'k');
hold on;
stem(t_s(2,:),x_s(2,:),'r','LineWidth',1.5)
title('x(t) and random points 2.d')
xlabel('t[sec]','FontSize',10)
ylabel('[V]','FontSize',10)
legend('x(t)','xrandom_s(t)')

figure
plot(t,x_t, 'k')
hold on
plot(t,x_res(2,:),'--b', 'LineWidth',2)
legend('x(t)', 'xrandom_s(t)(t)'); 
title('Signal x(t) restored 2.d')
xlabel('t[sec]','FontSize',10);
ylabel('[V]','FontSize',10)


%2.e
%%
M = 5;
wb = 2*pi;
wa = 5*pi;
T = 2;
w0 = 2*pi/T;
nsmpl = 11;
t_s(2,:) = T*rand([1,nsmpl]);
x_s(2,:) = 5*cos(wa*t_s(2,:))-3*sin(wb*t_s(2,:));
x_res = zeros(size(t_s,1),length(t));
x_t = 5*cos(wa*t)-3*sin(wb*t);
F = zeros(nsmpl,M);
for n = 1:nsmpl
   for m = 1:2*M+1
       F(n,m) = exp(1i*(t_s(2,n)+(0.01*rand(1)))*(-5+m-1)*w0) ;  
   end
end
if nsmpl == 2*M+1
   a(2,:) = inv(F)*x_s(2,:)';
elseif nsmpl>2*M+1
   a(2,:) = inv(F'*F)*F'*x_s(2,:)';
end
cond_num(3) = cond(F);

for k = 1:2*M+1
    x_res(2,:) = x_res(2,:) + a(2,k)*exp(1i*(k-M-1)*w0*t);
end  
cond_num(3)

figure
plot(t,x_t,'k');
hold on;
stem(t_s(2,:),x_s(2,:),'r','LineWidth',1.5)
title('x(t) and random points 2.e')
xlabel('t[sec]','FontSize',10)
ylabel('[V]','FontSize',10)
legend('x(t)','xrandom_s(t)')

figure
plot(t,x_t, 'k')
hold on
plot(t,x_res(2,:),'--b', 'LineWidth',2)
legend('x(t)', 'xrandom_s(t)(t)'); 
title('Signal x(t) restored 2.e')
xlabel('t[sec]','FontSize',10);
ylabel('[V]','FontSize',10)

%2.f
%%

n_sample_40 = 40;
t_s_40 = T*rand([1,n_sample_40]);
x_res_40 = zeros(1,length(t));
a_40 = zeros(1,n_sample_40);

x_s_40 = 5*cos(wa*t_s_40)-3*sin(wb*t_s_40);
F_40 = zeros(n_sample_40,M);
for n = 1:n_sample_40
    for m = 1:2*M+1
        F_40(n,m) = exp(1i*(t_s_40(n)+0.01*rand)*(-5+m-1)*w0);
    end
end

if n_sample_40 == 2*M+1
    a_40 = inv(F_40)*x_s_40';
elseif n_sample_40>2*M+1
    a_40 = inv(F_40'*F_40)*F_40'*x_s_40';
end 
cond_num_40 = cond(F_40);
for k = 1:2*M+1
    x_res_40 = x_res_40 + a_40(k)*exp(1i*(k-M-1)*w0*t);
end
a_40(:)
figure
plot(t,x_t,'k');
hold on;
stem(t_s_40,x_s_40,'b','LineWidth',1.5)
title('Signal x(t) 40 points 2.f')
xlabel('t[sec]','FontSize',12)
ylabel('Amplitude[V]','FontSize',12)
legend('x(t)','xrand_s(t)')

figure
plot(t,x_t)
hold on
plot(t,x_res_40,'--k', 'LineWidth',2)
legend('x(t)', 'xrand_r_e_s(t)'); title('Signal x(t) 40 points restored 2.f')
xlabel('t[sec]','FontSize',12); ylabel('Amplitude[V]','FontSize',12)

%3.a
%%

T = 10;
Nphi = -20:20;
t = 0:1/100:T;
Npsi = 0:19;
c_psi_f = Trapez_Method(f_t, psi_matrix, T);
c_psi_g = Trapez_Method(g_t, psi_matrix, T);
[phi_limits, time_phi] = meshgrid(Nphi, t);
phi_matrix = exp(((1i*2*pi)/T) .* phi_limits .* time_phi);
f_t = 4*cos(((4*pi)/T).*t) + sin(((10*pi)/T).*t);
g_t = 2*sign(sin(((6*pi)/T).*t)) - 4*sign(sin(((4*pi)/T).*t));
c_phi_f = Trapez_Method(f_t, phi_matrix, T);
c_phi_g = Trapez_Method(g_t, phi_matrix, T);
[psi_limits, time_psi] = meshgrid(Npsi, t);
psi_matrix = rectangularPulse(time_psi.*(20/T)-(psi_limits+0.5));


f_phi_res = sum(c_phi_f.*phi_matrix,2);
g_phi_res = sum(c_phi_g.*phi_matrix,2);
f_psi_res = sum(c_psi_f.*psi_matrix,2);
g_psi_res = sum(c_psi_g.*psi_matrix,2);

figure
subplot(2,1,1);
plot(t,f_t,'k');
hold on;
plot(t,f_phi_res,'--r','LineWidth',0.05)
plot(t,f_psi_res,'--b','LineWidth',0.05)
title('Signal f(t) and restored f^~ 3.c')
xlabel('t[sec]','FontSize',12)
ylabel('[V]','FontSize',12)
legend('f(t)','f^~_\phi(t)','f^~_\psi(t)')

subplot(2,1,2);
plot(t,g_t,'k');
hold on;
plot(t,g_phi_res,'--r','LineWidth',0.5)
plot(t,g_psi_res,'--b','LineWidth',0.5)
title('g(t) and restored g^~ 3.c')
xlabel('t[sec]','FontSize',10)
ylabel('[V]','FontSize',10)
legend('g(t)','g^~_\phi(t)','g^~ 3.c')


function [c] = Trapez_Method(x_t, twoDArray, T)
    rangeVal = linspace(0,T,length(x_t));
    c = zeros(1,size(twoDArray,2));
    for n = 1:size(twoDArray,2)
        bottom = trapz(rangeVal, twoDArray(:,n).*conj(twoDArray(:,n)));
        top = trapz(rangeVal, x_t'.*conj(twoDArray(:,n)));
        if (bottom~=0) , c(n) = top/bottom;
        else ,c(n) = 0;
        end
    end
end


