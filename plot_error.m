function plot_error(dataset_name)
	
	clf;
	error_train = strcat(dataset_name, "_errorTrain.txt");
	error_test = strcat(dataset_name, "_errorTest.txt");

	errorTrain = load(error_train);
	errorTest = load(error_test);

	hold on;
	plot(errorTrain(:,1), "b", "linewidth", 3);
	plot(errorTest(:,1), "r", "linewidth", 3);

	grid on;
	title(dataset_name, "fontsize", 15);
	set(gca, "fontsize", 15);
	xlabel("Epocas");
	ylabel("Erro Medio");
	legend("Train", "Test", "location", "northeast");

	file_out = strcat(dataset_name, ".png");
	print(file_out, "-dpng", "-r0", "-color", "-S800,600");

endfunction
