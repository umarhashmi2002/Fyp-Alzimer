import React, { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('0');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.mgz')) {
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError('Please upload a valid MRI image file (.mgz format).');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    if (!file) {
      setError('Please upload an MRI image file.');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('age', age);
    formData.append('sex', sex);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data.prediction);
      setError(null);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.error || 'Server error occurred.');
      } else if (error.request) {
        setError('No response received from the server. Please try again later.');
      } else {
        setError('An unknown error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 5 }}>
      <Card>
        <CardContent>
          <Typography variant="h5" component="div" gutterBottom>
            MRI Alzheimer Prediction
          </Typography>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <form onSubmit={handleSubmit} noValidate>
            <FormControl fullWidth margin="normal">
              <InputLabel shrink>Upload MRI Image</InputLabel>
              <input
                type="file"
                accept=".mgz"
                onChange={handleFileChange}
                style={{
                  marginTop: 10,
                  marginBottom: 20,
                }}
              />
            </FormControl>
            <TextField
              fullWidth
              margin="normal"
              label="Age"
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              required
            />
            <FormControl fullWidth margin="normal">
              <InputLabel id="sex-select-label">Sex</InputLabel>
              <Select
                labelId="sex-select-label"
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                required
              >
                <MenuItem value="0">Male</MenuItem>
                <MenuItem value="1">Female</MenuItem>
              </Select>
            </FormControl>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              sx={{ mt: 2 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Submit'}
            </Button>
          </form>
        </CardContent>
      </Card>
      {result && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6">Prediction Result:</Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Class</TableCell>
                    <TableCell align="right">Probability (%)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.probabilities && result.probabilities[0].map((probability, index) => (
                    <TableRow key={index}>
                      <TableCell>{`Class ${index + 1}`}</TableCell>
                      <TableCell align="right">{(probability * 100).toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default UploadForm;
