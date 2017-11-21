"""
Test property parsing of non-loose cases with and without keywords.
Test case is 10 property values; keyword is 'E0', like in VASP OSZICAR files.
"""
import unittest
import numpy as np
from representation import parse_property


class PropertyParseTestCase(unittest.TestCase):
    def setUp(self):
        """
        Generate text with embedded property values for both cases.
        """
        n = 10
        self.keyword = 'E0'
        self.props = np.subtract(np.multiply(np.random.rand(n, ),
                                             10000), 5000)
        # random property values, between -5000 and 5000

        keyword_joiners = ['=', ':', ',']
        keyword_separators = [';', ',', '   ', '\n']
        randomtext = ['asd1 ;2fg\nh632,jk\nlq12w  eryui', 'sdf5\nasdf,ds',
                      'ui=bvc', 'eruio,bs']
        separators = ['\n', ',', ';', ' ']
        # property values (jumbled .csv-like)
        self.text_l = ''.join([str(prop) + np.random.choice(separators)
                               for prop in self.props])

        # property values (jumbled text with keywords)
        self.text_kw = ''.join([str(self.keyword) + ' ' * np.random.randint(5)
                                + np.random.choice(keyword_joiners)
                                + ' ' * np.random.randint(5) + str(prop)
                                + np.random.choice(keyword_separators)
                                + np.random.choice(randomtext)
                                for prop in self.props])

    def test_parse_property(self):
        self.parsed = np.asarray(parse_property(self.text_l))
        self.assertTrue(np.all(np.isclose(self.parsed, self.props, atol=1e-4)))

    def test_parse_property_kw(self):
        self.parsed = np.asarray(parse_property(self.text_kw,
                                                keyword=self.keyword))
        self.assertTrue(np.all(np.isclose(self.parsed, self.props, atol=1e-4)))


if __name__ == '__main__':
    unittest.main()
