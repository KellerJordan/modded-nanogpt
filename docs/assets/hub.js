// docs/assets/hub.js
(async () => {
  try {
    // Update the repository name to match the actual repository
    const csvUrl = 'https://raw.githubusercontent.com/Synthyra/SpeedrunningPLMs/main/data/experiments.csv';
    
    console.log('Attempting to fetch CSV from:', csvUrl);

    // Fetch & parse
    const response = await fetch(csvUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const csvText = await response.text();
    console.log('CSV data received:', csvText.slice(0, 200) + '...');
    
    const { data, meta } = Papa.parse(csvText, { header: true, skipEmptyLines: true });
    console.log('Parsed data:', data);
    console.log('Meta fields:', meta.fields);

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
    
    console.log('DataTable initialized successfully');
    
  } catch (error) {
    console.error('Error loading or processing data:', error);
    
    // Display error message to user
    const tableContainer = document.getElementById('exp-table');
    if (tableContainer) {
      tableContainer.innerHTML = `
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded" role="alert">
          <strong class="font-bold">Error loading data!</strong>
          <span class="block sm:inline"> ${error.message}</span>
          <details class="mt-2">
            <summary class="cursor-pointer">Technical details</summary>
            <pre class="mt-2 text-xs">${error.stack}</pre>
          </details>
        </div>
      `;
    }
  }
})();
