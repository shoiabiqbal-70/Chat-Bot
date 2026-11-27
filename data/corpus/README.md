# Public evaluation corpus

- `Energy_Literacy_text_only_3.0.docx` and `Energy_Literacy_Low_Res_3.0.pdf`: [U.S. Department of Energy, Energy Literacy Framework](https://www.energy.gov/cmei/education/articles/download-copy-energy-literacy-framework), version 3.0 (2014).
- `energy-saver-guide-2022.pdf`: [U.S. Department of Energy, Energy Saver Guide](https://www.energy.gov/sites/default/files/2022-08/energy-saver-guide-2022.pdf), 2022.

The PDF and DOCX versions of the Energy Literacy Framework contain overlapping content and should not both be indexed for the two-document evaluation. The fixed evaluation corpus uses the DOCX and the distinct Energy Saver PDF. The PDF copy remains here to validate page-aware extraction and PDF upload.

The Energy Saver PDF extracts into 54 pages and 108,251 characters with the current `pypdf` code. Some fonts trigger warnings and a few extracted characters are garbled; inspect source pages when judging answer quality.
