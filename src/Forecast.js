import React, { useState } from 'react';
import Papa from 'papaparse';
import './Forecast.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import * as tf from '@tensorflow/tfjs';

const Forecast = () => {
  const [data, setData] = useState([]);
  const [months, setMonths] = useState(6);  // Default to 6 months
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState('');
  const [monthsError, setMonthsError] = useState('');
  const [fileError, setFileError] = useState('');
  const [newProduct, setNewProduct] = useState({ description: '', totalSold: '', created: '' });
  const [editingProductIndex, setEditingProductIndex] = useState(null); // Track the product being edited

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type !== 'text/csv') {
        setFileError('Invalid file type. Please upload a CSV file.');
        setTimeout(() => setFileError(''), 5000);
        return;
      } else {
        setFileError('');
      }
  
      Papa.parse(file, {
        complete: (result) => {
          const processedData = preprocessData(result.data);
          if (processedData.length === 0) {
            setFileError('No valid data found in the CSV file.');
            setTimeout(() => setFileError(''), 5000);
          } else {
            setData(processedData);
          }
        },
        header: true,
      });
    } else {
      setFileError('No file selected.');
      setTimeout(() => setFileError(''), 3000);
    }
  };

  // Data preprocessing function
  const preprocessData = (rawData) => {
    const limitData = Object.values(rawData).slice(0, 100);
    const filteredData = limitData.filter(
      (row) => row.created && row.short_desc && row.total_sold && isNaN(row.short_desc)
    );
    const result = filteredData
      .map((entry) => {
        if (!entry.created || !entry.short_desc || !entry.total_sold) return null;
        const dateParts = entry.created.split('-');
        if (dateParts.length < 2) return null;
        const month = parseInt(dateParts[1], 10);
        return { month, product: entry.short_desc, quantity: parseFloat(entry.total_sold) };
      })
      .filter((entry) => entry !== null);

    return result;
  };

  // Prepare data for model
  const prepareData = (productData) => {
    const months = productData.map((entry) => entry.month);
    const quantities = productData.map((entry) => entry.quantity);

    const xs = tf.tensor2d(months.map((month, index) => [month, index]));
    const ys = tf.tensor1d(quantities);

    return { xs, ys };
  };

  // Function to predict sales for a specific product
  const predictSalesForProduct = async (productData) => {
    try {
      const { xs, ys } = prepareData(productData);

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 64, inputShape: [2], activation: 'relu' }));
      model.add(tf.layers.dense({ units: 1 }));

      model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

      await model.fit(xs, ys, { epochs: 10 });

      const predictions = [];
      const lastMonth = productData[productData.length - 1].month;

      for (let i = 1; i <= months; i++) {
        const inputTensor = tf.tensor2d([[lastMonth + i, productData.length]]);
        const predictedQuantity = model.predict(inputTensor).dataSync()[0];
        predictions.push({ month: lastMonth + i, predicted: predictedQuantity });
      }

      return predictions;
    } catch (error) {
      setFileError('Error during model training or prediction: ' + error.message);
      setTimeout(() => setFileError(''), 3000);
      console.error(error);
      return [];
    }
  };

  // Function to predict sales
  const predictSales = async () => {
    if (!data || data.length === 0) {
      setFileError('Please upload a CSV file first or add a product data first.');
      setTimeout(() => setFileError(''), 3000);
      return;
    }

    if (months <= 0) {
      setMonthsError('Please enter a positive number for months.');
      // Clear the error after 5 seconds
      setTimeout(() => setMonthsError(''), 5000);
      return;
    } else {
      setMonthsError('');
    }
    setLoading(true);

    const productPredictions = {};

    const uniqueProducts = [...new Set(data.map((entry) => entry.product))];

    for (let product of uniqueProducts) {
      const filteredData = data.filter((entry) => entry.product === product);

      if (filteredData.length === 0) {
        setFileError(`No data found for the product: ${product}`);
        setTimeout(() => setFileError(''), 3000);
        continue;
      }

      const predictions = await predictSalesForProduct(filteredData);
      productPredictions[product] = predictions;
    }

    setPredictions(productPredictions);
    setLoading(false);
  };

  // Handle product selection
  const handleProductChange = (event) => {
    setSelectedProduct(event.target.value);
  };

  // Handle input changes for the new product
  const handleInputChange = (e) => {
    setNewProduct({ ...newProduct, [e.target.name]: e.target.value });
  };

  // Add a new product to the data
  const addProduct = () => {
    if (!newProduct.description || !newProduct.totalSold || !newProduct.created) {
      setFileError('Please fill in all fields.');
      setTimeout(() => setFileError(''), 3000);
      return;
    }
    const dateParts = newProduct.created.split('-');
    if (dateParts.length < 2 || isNaN(newProduct.totalSold)) {
      setFileError('Invalid date format or total sold value.');
      setTimeout(() => setFileError(''), 3000);
      return;
    }

    const newEntry = {
      product: newProduct.description,
      quantity: parseFloat(newProduct.totalSold),
      created: newProduct.created,
      month: parseInt(dateParts[1], 10),
    };

    setData([...data, newEntry]);
    setNewProduct({ description: '', totalSold: '', created: '' });
  };

  // Start editing a product
  const editProduct = (index) => {
    const productToEdit = data[index];
    setNewProduct({ description: productToEdit.product, totalSold: productToEdit.quantity, created: productToEdit.created });
    setEditingProductIndex(index); // Set the index of the product being edited
  };  

  // Save the edited product
  const saveEditedProduct = () => {
    if (!newProduct.description || !newProduct.totalSold || !newProduct.created) {
      setFileError('Please fill in all fields.');
      setTimeout(() => setFileError(''), 3000);
      return;
    }
  
    const updatedProduct = {
      product: newProduct.description,
      quantity: parseFloat(newProduct.totalSold),
      created: newProduct.created,
      month: parseInt(newProduct.created.split('-')[1], 10),
    };
  
    const updatedData = [...data];
    updatedData[editingProductIndex] = updatedProduct; // Update the specific product being edited
    setData(updatedData);
    setNewProduct({ description: '', totalSold: '', created: '' });
    setEditingProductIndex(null); // Reset editing state
  };

  const cancelEdit = () => {
    setNewProduct({ description: '', totalSold: '', created: '' });
    setEditingProductIndex(null); // Cancel editing
  };  

  // Delete a product from the data
  const deleteProduct = (index) => {
    const updatedData = data.filter((_, i) => i !== index);
    setData(updatedData);
  };

  const renderChartData = (product) => {
    const chartData = [];
    const filteredData = data.filter((entry) => entry.product === product);
  
    filteredData.forEach((entry) => {
      const predictedEntry = predictions[product]?.find((pred) => pred.month === entry.month);
      let month = entry.month;
      let year = 2023; // Default year
  
      // Adjust for months greater than 12 (reset month after 12)
      if (month > 12) {
        const adjustedMonth = month % 12 || 12; // Reset to 01 after December
        const adjustedYear = year + Math.floor(month / 12); // Increment the year after December
        chartData.push({
          month: adjustedMonth,
          monthYear: `${adjustedMonth < 10 ? '0' : ''}${adjustedMonth}/${adjustedYear}`, // Updated month and year
          actual: entry.quantity,
          predicted: predictedEntry ? predictedEntry.predicted : null,
        });
      } else {
        chartData.push({
          month: entry.month,
          monthYear: `${entry.month < 10 ? '0' : ''}${entry.month}/2023`, // Standard year formatting
          actual: entry.quantity,
          predicted: predictedEntry ? predictedEntry.predicted : null,
        });
      }
    });
  
    predictions[product]?.forEach((pred) => {
      let month = pred.month;
      let year = 2023; // Default year
  
      // Adjust for months greater than 12
      if (month > 12) {
        const adjustedMonth = month % 12 || 12; // Reset to 01 after December
        const adjustedYear = year + Math.floor(month / 12); // Increment the year after December
        chartData.push({
          month: adjustedMonth,
          monthYear: `${adjustedMonth < 10 ? '0' : ''}${adjustedMonth}/${adjustedYear}`, // Updated month and year
          actual: null,
          predicted: pred.predicted,
        });
      } else {
        chartData.push({
          month: pred.month,
          monthYear: `${pred.month < 10 ? '0' : ''}${pred.month}/2023`, // Standard year formatting
          actual: null,
          predicted: pred.predicted,
        });
      }
    });
  
    return chartData;
  };

  return (
    <div className="forecast-container">
      <h1>Sales Forecast</h1>

      <div className="main-content">
        <div className="product-data">
          {/* Product Selection, File Upload, and Months Input */}
          <div className="product-selector">
            <label htmlFor="product">Select Product: </label>
            <select id="product" value={selectedProduct} onChange={handleProductChange}>
              <option value="">Select a Product</option>
              {[...new Set(data.map((entry) => entry.product))].map((product) => (
                <option key={product} value={product}>
                  {product}
                </option>
              ))}
            </select>
          </div>

          <div className="file-upload">
            <input type="file" accept=".csv" onChange={handleFileUpload} />
            {fileError && <div className="error">{fileError}</div>}
          </div>

          <div className="months-selector">
            <label htmlFor="months">Number of months to predict: </label>
            <input
              type="number"
              id="months"
              value={months}
              onChange={(e) => {
                const value = Number(e.target.value);
                setMonths(value);
                if (value <= 0) setMonthsError('Please enter a positive number for months.');
                else setMonthsError('');
              }}
              min="1"
              style={{ padding: '8px', fontSize: '16px', width: '120px', marginBottom: '10px' }}
            />
            {monthsError && <p style={{ color: 'red', fontSize: '14px' }}>{monthsError}</p>}
          </div>

          <div className="predict-button-container">
            <button 
              onClick={predictSales} 
              disabled={loading} 
              style={{ marginBottom: '10px' }}
              className="predict-button">
              {loading ? 'Predicting...' : `Predict Sales`}
            </button>
          </div>

          <ResponsiveContainer width="100%" height={300}>
          <LineChart data={renderChartData(selectedProduct)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="monthYear" 
              label={{ value: 'Month', position: 'insideBottom', offset: -10 }} 
              tick={{ fontSize: 12 }} 
            />
            <YAxis label={{ value: 'Quantity', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value) => `${value} units`} />
            <Legend wrapperStyle={{ paddingTop: 30 }} />
            <Line type="monotone" dataKey="actual" stroke="#8884d8" style={{ marginBottom: '-10px' }} />
            <Line type="monotone" dataKey="predicted" stroke="#82ca9d" />
          </LineChart>
        </ResponsiveContainer>

        </div>

        {/* Product CRUD Operations */}
        <div className="product-crud-container" style={{ marginTop: '20px' }}>
          <div className="product-crud">
            <h3>{editingProductIndex !== null ? 'Edit Product' : 'Add New Product'}</h3>
            <div className="product-crud-form">
              <input
                className="input-field"
                type="text"
                name="description"
                placeholder="Product Description"
                value={newProduct.description}
                onChange={handleInputChange}
              />
              <input
                className="input-field"
                type="number"
                name="totalSold"
                placeholder="Total Sold"
                value={newProduct.totalSold}
                onChange={handleInputChange}
              />
              <input
                className="input-field input-date"
                type="date"
                name="created"
                value={newProduct.created}
                onChange={handleInputChange}
              />
              {editingProductIndex !== null ? (
                <div>
                  <button onClick={saveEditedProduct}>Save Changes</button>
                  <button onClick={cancelEdit}>Cancel</button>  {/* Cancel button */}
                </div>
              ) : (
                <button onClick={addProduct}>Add Product</button>
              )}
            </div>
          </div>

          {/* Product List with CRUD Operations */}
          <ul className="product-list" style={{ maxHeight: '300px', overflowY: 'scroll' }}>
            {data.map((entry, index) => (
              <li key={index} className="product-item">
                <div className="product-details">
                  {entry.product} - {entry.quantity} units sold - {entry.created}
                </div>
                <div className="product-buttons">
                  <button onClick={() => editProduct(index)}>Edit</button>
                  <button onClick={() => deleteProduct(index)}>Delete</button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Forecast;
