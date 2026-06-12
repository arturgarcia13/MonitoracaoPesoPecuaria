import unittest
import numpy as np
from scripts.simulacao import simulate_data

class TestSimulacao(unittest.TestCase):

    def test_simulate_data_output_shape(self):
        """Testa se a função retorna o número correto de amostras e colunas."""
        n_samples = 100
        df = simulate_data(n_samples=n_samples, seed=42)
        
        self.assertEqual(len(df), n_samples)
        self.assertEqual(len(df.columns), 7)
        
        expected_cols = [
            'Idade_dias', 'PN_kg', 'Brix_perc', 'Isolamento_dias',
            'Sexo_Macho', 'Filho_Novilha', 'Peso_kg'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_simulate_data_values_range(self):
        """Testa se os valores gerados estão dentro dos limites biológicos esperados."""
        df = simulate_data(n_samples=500, seed=42)
        
        # Idade entre 1 e 180
        self.assertGreaterEqual(df['Idade_dias'].min(), 1)
        self.assertLessEqual(df['Idade_dias'].max(), 180)
        
        # PN entre 18 e 45 kg
        self.assertGreaterEqual(df['PN_kg'].min(), 18.0)
        self.assertLessEqual(df['PN_kg'].max(), 45.0)
        
        # Brix entre 12 e 38
        self.assertGreaterEqual(df['Brix_perc'].min(), 12.0)
        self.assertLessEqual(df['Brix_perc'].max(), 38.0)
        
        # Variáveis binárias
        self.assertTrue(df['Sexo_Macho'].isin([0, 1]).all())
        self.assertTrue(df['Filho_Novilha'].isin([0, 1]).all())
        
        # Isolamento (Poisson) >= 0
        self.assertGreaterEqual(df['Isolamento_dias'].min(), 0)
        
        # Peso deve ser positivo e razoável
        self.assertGreater(df['Peso_kg'].min(), 10.0)

    def test_simulate_data_reproducibility(self):
        """Testa se a semente aleatória garante a reprodutibilidade."""
        df1 = simulate_data(n_samples=50, seed=123)
        df2 = simulate_data(n_samples=50, seed=123)
        
        np.testing.assert_array_equal(df1['Peso_kg'].values, df2['Peso_kg'].values)

if __name__ == '__main__':
    unittest.main()
