document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('load-data-button').addEventListener('click', loadStudentData);

    async function loadStudentData() {
        const studentIdInput = document.getElementById('student-input');
        const enteredStudentId = studentIdInput.value;

        try {
            const data = await fetchDataFromServer(enteredStudentId);

            document.getElementById('student-id').textContent = data.student_id;
            document.getElementById('attention-span-subject-1').textContent = data.subjects[0].attention_span;

            const lineChartCanvas = document.getElementById('line-chart');
            updateLineChart(lineChartCanvas, data.subjects);
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }

    function fetchDataFromServer(studentId) {
        // Replace this with actual data fetching logic from your server
        // Example: return fetch(`/api/student/${studentId}`).then(response => response.json());
        
        // Dummy data for testing
        return {
            student_id: studentId,
            subjects: [
                { name: 'Distributed Computing', attention_span: getRandomAttentionSpan() },
            ],
        };
    }

    function updateLineChart(canvas, subjects) {
        new Chart(canvas, {
            type: 'line',
            data: {
                labels: ['5/12/23', '6/12/23', '7/12/23', '8/12/23', '9/12/23'],
                datasets: subjects.map((subject, index) => ({
                    label: subject.name,
                    borderColor: getBorderColor(index),
                    data: generateRandomData(),
                    fill: false,
                })),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        labels: ['5/12/23', '6/12/23', '7/12/23', '8/12/23', '9/12/23'],
                    },
                    y: {
                        beginAtZero: true,
                    },
                },
            },
        });
    }

    function getBorderColor(index) {
        const colors = ['rgb(75, 192, 192)'];
        return colors[index % colors.length];
    }

    function getRandomAttentionSpan() {
        // Generate a random attention span between 50 and 100
        return Math.random() * 50 + 50;
    }

    function generateRandomData() {
        // Generate an array of random data for testing
        return Array.from({ length: 7 }, () => Math.floor(Math.random() * 50) + 50);
    }
});

