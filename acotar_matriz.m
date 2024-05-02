function matriz_acotada = acotar_matriz(matriz, limite_inferior, limite_superior)
    % Acotar los valores de la matriz
    matriz_acotada = matriz;
    matriz_acotada(matriz_acotada < limite_inferior) = limite_inferior;
    matriz_acotada(matriz_acotada > limite_superior) = limite_superior;
end