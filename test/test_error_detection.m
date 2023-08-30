mix_exp_SS_y = struct('p', [0.5; 0.3; 0.2], 'lambda', [1; 2; 4]);

rng(1);
drawState(12);
drawObs(123);
hessianMethod(1234);

clear x;
x.N = 100;
x.mu = -9;
x.phi = 0.95;
x.omega = 20;

model = 'mix_exp_SS';
theta = struct('x', x, 'y', mix_exp_SS_y);

% Initial draw (y,x)
x = drawState(theta)
"hello1"
y = drawObs(x, model, theta)
"hello2"
hmout = hessianMethod('mix_exp_SS', y, theta, 'EvalAtState', x)
"hello3"