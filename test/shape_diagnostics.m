function shape_diagnostics(th1, sh1, th2, sh2)
	delta1 = th1 - th2;
	v1_predict_1 = sh2.v + sh2.g' * delta1;
	v1_predict_2 = v1_predict_1 + 0.5 * delta1' * sh2.H * delta1;
	g1_predict = sh2.g + sh2.H * delta1;

	fprintf('Values: at th1, %f, at th2, %f\n', sh1.v, sh2.v);

	fprintf('For value at th1, error of order 0: %f, order 1: %f, order 2: %f\n', ...
		sh2.v - sh1.v, v1_predict_1 - sh1.v, v1_predict_2 - sh1.v);

	delta2 = -delta1;
	v2_predict_1 = sh1.v + sh1.g' * delta2;
	v2_predict_2 = v2_predict_1 + 0.5 * delta2' * sh1.H * delta2;
	g2_predict = sh1.g + sh1.H * delta2;

	fprintf('For value at th2, error of order 0: %f, order 1: %f, order 2: %f\n', ...
		sh1.v - sh2.v, v2_predict_1 - sh2.v, v2_predict_2 - sh2.v);

	display(sh1.g, 'Gradient at th1');
	display(sh2.g, 'Gradient at th2')
	display(g1_predict, '1st order prediction of gradient at th1');
	display(g2_predict, '1st order prediction of gradient at th2');
	display(g1_predict - sh1.g, 'Gradient prediction error at th1 from th2')
	display(g2_predict - sh2.g, 'Gradient prediction error at th2 from th1')
end
