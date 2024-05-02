function [digit]=print_digit(digits,num)
    digit = zeros(28, 28); % Initialize the digit matrix
    for i = 1:28
        for j = 1:28
            digit(i, j) = digits((i - 1) * 28 + j, num);
        end
    end
    %imshow(digit); %Display the digit
end