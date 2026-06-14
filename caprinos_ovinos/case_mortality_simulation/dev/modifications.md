# alterações

- tornar gmd_base em função do sexo
- tornar p_ref_gmd em função do sexo e unir com o p_opt da curva de risco, para ter um único parâmetro que defina o peso ótimo de nascimento.
- adicionar beta_trigemeos na forma do beta_parto, para ter uma penalidade específica para trigêmeos.
- modificar beta0 em função do sexo, para ajustar a mortalidade geral.
- adicionar teta-z, z-score_atual - z_score_nasc: z-score_atual = (peso_atual-peso_esperado)/desvio_esperado.
- desvio esperado t: desvio_base * (1 + gamma * (t/t_max)**2)
- risco_t = prob_mortalidade_nascimento * e^-(k * teta_z)