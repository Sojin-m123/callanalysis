document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('form.form-section');
  const spinner = document.getElementById('loading-spinner');
  const rightPanel = document.getElementById('right-panel');

  // Show loading spinner and scroll up on form submit
  if (form && spinner && rightPanel) {
    form.addEventListener('submit', () => {
      spinner.classList.remove('d-none');
      rightPanel.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Plot sentiment comparison chart
  if (window.resultData) {
    const mlSentiments = resultData.ml_sentiments || [];
    const enSentiments = resultData.en_sentiments || [];
    const labels = ['positive', 'neutral', 'negative'];

    const count = (arr) =>
      labels.map((label) => arr.filter((s) => s === label).length);

    Plotly.newPlot('sentimentChart', [
      {
        x: labels,
        y: count(mlSentiments),
        name: 'Malayalam',
        type: 'bar',
        marker: { color: '#42a5f5' },
      },
      {
        x: labels,
        y: count(enSentiments),
        name: 'English',
        type: 'bar',
        marker: { color: '#ef5350' },
      },
    ], {
      barmode: 'group',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#fff' }
    });
  }
});
document.addEventListener('DOMContentLoaded', function () {
  const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
  tabs.forEach(tab => {
    tab.addEventListener('shown.bs.tab', function (e) {
      setTimeout(() => {
        Plotly.Plots.resize('weeklyLeadBar');
        Plotly.Plots.resize('categoryPie');
        Plotly.Plots.resize('highInterestBar');
      }, 300);  // ⏳ Delay ensures the div is visible
    });
  });
});
$(document).ready(function () {
  $('#lead-categories').select2({
    placeholder: "Select Lead Categories",
    tags: false,
    width: '100%',
    allowClear: true,
    theme: "bootstrap-5"
  });
});
