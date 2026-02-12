

field_gradient = linspace(200,500,2); %V/cm
Offset_to_ground = 1000*ones(size(field_gradient)); % V;
lens_VMI = 1.38*ones(size(field_gradient));


I_grid = -800/120*field_gradient + Offset_to_ground;
VMI2 = zeros(size(field_gradient));
VMI1 = zeros(size(field_gradient));
e_grid = zeros(size(field_gradient));
dt_e = zeros(size(field_gradient));
for idx = 1:length(field_gradient)

    while I_grid(idx) ~= 0
        I_grid(idx) = -800/120*field_gradient(idx) + Offset_to_ground(idx);
        VMI2(idx) = Offset_to_ground(idx);
        VMI1(idx) = VMI2(idx)-(VMI2(idx) - I_grid(idx))/lens_VMI(idx);
        e_grid(idx) = VMI1(idx) + 0.5*(VMI2(idx)-VMI1(idx));
        dt_e(idx) = VMI2(idx);
        Offset_to_ground(idx) = Offset_to_ground(idx) - I_grid(idx);
    end
end


parameters.field_gradient = field_gradient;
parameters.Offset_to_ground = Offset_to_ground;
parameters.lens_VMI = lens_VMI;
parameters.I_grid = I_grid;
parameters.VMI2 = VMI2;
parameters.VMI1 = VMI1;
parameters.e_grid = e_grid;
parameters.dt_e = dt_e;


save parameters parameters
