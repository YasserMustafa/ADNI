conf = cell(1, numel(pet_path));

for i=1:numel(pet_path)
    pet = cell2mat(pet_path{i}')';
    mmse = cell2mat(mmse_path{i}')';

    conf{i} = normalize(confusionmat(pet, mmse), 2);
end

cat_conf = cat(3, conf{:});
mean_conf = mean(cat_conf, 3);
std_conf = std(cat_conf, 0, 3);