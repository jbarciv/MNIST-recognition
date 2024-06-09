% Epoch = 3;
layers = [7 11 15 19];

success_training = [96.775 97.212 97.9 98.475];
success_testing  = [94.3 96 96.8 97.25];

figure();
subplot(3,1,1)
plot(layers,success_training,'b--')
hold on; grid on;
plot(layers,success_testing,'b')
xlabel('Number of Layers')
ylabel('% of Success')
legend('training','testing','Location','best')
title('Layers Evolution (Epoch=3)')
ylim([92 100])

% Layer = 11; epoch = 3;

conv_layer1 = [1 2 4 8 16];
conv_layer2 = 2*conv_layer1;

success_training = [90.95 93.33 95.94 97.22 98.71];
success_testing  = [90.4 91.65 94.45 96 97];

subplot(3,1,2)
plot(conv_layer1,success_training,'b--')
hold on; grid on;
plot(conv_layer1,success_testing,'b')
ax2.XAxisLocation = 'top';
xlabel('Number of Neurons in Layer1')
ylabel('% of Success')
legend('training','testing','Location','best')
title('Neurons Evolution (Layers = 11; Epoch=3)')
ylim([88 100])
xlim([0 17])

epoch = [1 2 3 4];
success_training = [90.56 92.925 93.3 95.063];
success_testing  = [90.3 91.4 92.35 93.7];

subplot(3,1,3)
plot(epoch,success_training,'b--')
hold on; grid on;
plot(epoch,success_testing,'b')
xlabel('Number of Epochs')
ylabel('% of Success')
legend('training','testing','Location','best')
title('Epoch Evolution (Layers = 11; Layer1 Neurons=2)')
ylim([88 100])
xlim([1 4])
