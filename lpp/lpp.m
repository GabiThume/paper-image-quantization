%	Author: Tiago Nazare
%	Universidade de São Paulo / ICMC / 2014
 
 function tests()
    files = dir('originals2/*.csv');
    for file = files'
        %csv = load(file.name);
        %disp(file);
        file_path = strcat('originals2/', file.name);
        test_one(file_path);
    end
end

function test_one(filename)
    %disp(filename);
    aux = strrep(filename, '.csv', '');
    aux = strrep(aux, 'originals2/', '');
    aux = strcat('lpp2/', aux);
    disp(aux);
    
    original = csvread(filename);
    
    in = original(:, 3:end)';
    
    out16 = lpp_he(in,10,16);
    out32 = lpp_he(in,10,32);
    out64 = lpp_he(in,10,64);
    out128 = lpp_he(in,10,128);
    
    write16 = [original(:,1:2) real(out16')];
    write32 = [original(:,1:2) real(out32')];
    write64 = [original(:,1:2) real(out64')];
    write128 = [original(:,1:2) real(out128')];
    
    csvwrite(strcat(aux, 'LPP-16.csv'), write16);
    csvwrite(strcat(aux, 'LPP-32.csv'), write32);
    csvwrite(strcat(aux, 'LPP-64.csv'), write64);
    csvwrite(strcat(aux, 'LPP-128.csv'), write128);
end
