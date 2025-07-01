// docs/assets/hub.js
(async () => {
    // Path to the raw CSV *on the same branch*; adjust if needed
    const csvUrl = 'https://raw.githubusercontent.com/<ORG>/<REPO>/main/data/experiments.csv';
  
    // Fetch & parse
    const response = await fetch(csvUrl);
    const csvText = await response.text();
    const { data, meta } = Papa.parse(csvText, { header: true, skipEmptyLines: true });
  
    // Build the column list for DataTables from CSV headers
    const columns = meta.fields.map(field => ({ title: field, data: field }));
  
    // Inject DataTable
    new DataTable('#exp-table', {
      data,
      columns,
      responsive: true,
      searchable: true,
      sortable: true,
      paging: true,
      pageLength: 25,
      className: 'stripe hover',
      // Optional: highlight good/bad results, etc.
      createdRow: (row, rowData) => {
        if (rowData.accuracy >= 0.90) row.classList.add('bg-green-50');
        if (rowData.failed === 'yes') row.classList.add('bg-red-50');
      },
    });
  })();
  