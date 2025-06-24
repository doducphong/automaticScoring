'use client';

import React, { useState, useRef } from 'react';
import {
  Box,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Text,
  Heading,
  Card,
  CardBody,
  useToast,
  Progress,
  Icon,
  Divider,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
} from '@chakra-ui/react';
import { FiUpload, FiCheck, FiFileText, FiFile } from 'react-icons/fi';
import api from '@/lib/api';

interface ExamResponse {
  id: string;
  score: number;
  student_name: string;
  exam_code: string;
  report: JSON;
  created_at: string;
}

export default function GradingPage() {
  // State for the DOCX tab
  const [docxFiles, setDocxFiles] = useState<File[]>([]);
  const [examTemplateFile, setExamTemplateFile] = useState<File | null>(null);
  const [xlsxFile, setXlsxFile] = useState<File | null>(null);
  const [xlsxTemplateFile, setXlsxTemplateFile] = useState<File | null>(null);
  const [docxStudentName, setDocxStudentName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [docxGradingResultsList, setDocxGradingResultsList] = useState<ExamResponse[]>([]);
  const [xlsxFiles, setXlsxFiles] = useState<File[]>([]);
  const [excelGradingResults, setExcelGradingResults] = useState<ExamResponse[]>([]);
  
  // File input references
  const docxFileInputRef = useRef<HTMLInputElement>(null);
  const examTemplateFileInputRef = useRef<HTMLInputElement>(null);
  const xlsxFileInputRef = useRef<HTMLInputElement>(null);
  const xlsxTemplateFileInputRef = useRef<HTMLInputElement>(null);
  
  const toast = useToast();

  // Handle file selection for DOCX
  const handleDocxFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setDocxFiles(Array.from(e.target.files));
    }
  };

  // Handle file selection for exam template
  const handleExamTemplateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setExamTemplateFile(e.target.files[0]);
    }
  };

  const handleXlsxFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setXlsxFile(e.target.files[0]);
    }
  };

  const parseReport = (report: string | object): any => {
    if (typeof report === 'string') {
      try {
        return JSON.parse(report);
      } catch {
        return null;
      }
    }
    return report;
  };

  const handleXlsxTemplateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setXlsxTemplateFile(e.target.files[0]);
    }
  };

  // Submit DOCX for grading
  const handleDocxSubmit = async () => {
    if (!examTemplateFile || docxFiles.length === 0) {
      toast({
        title: 'Missing files',
        description: 'Please select student DOCX files and a template file.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    setIsUploading(true);
    setDocxGradingResultsList([]);

    for (const file of docxFiles) {
      try {
        const formData = new FormData();
        formData.append('sample_file', examTemplateFile);
        formData.append('submission_file', file);
  
        const response = await api.post('/exams/grade-docx', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        let studentName = 'Unknown';
        if (response.data.student_id) {
          try {
            const studentRes = await api.get(`/students/${response.data.student_id}`);
            studentName = studentRes.data.name;
          } catch (err) {
            console.error("Error fetching student info:", err);
          }
        }
  
        const result: ExamResponse = {
          ...response.data,
          student_name: studentName,
        };
  
        setDocxGradingResultsList((prev) => [...prev, result]);
  
        toast({
          title: `Graded: ${file.name}`,
          description: `Score: ${result.score}`,
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      } catch (error) {
        console.error(`Error grading ${file.name}:`, error);
        toast({
          title: `Grading failed for ${file.name}`,
          status: 'error',
          duration: 4000,
          isClosable: true,
        });
      }
    }
  
    setIsUploading(false);
  };

  const handleXlsxSubmit = async () => {
    if (!xlsxTemplateFile || xlsxFiles.length === 0) {
      toast({
        title: 'Missing files',
        description: 'Please select both student Excel files and an exam template.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
  
    setIsUploading(true);
    setExcelGradingResults([]); // reset kết quả cũ
  
    for (const file of xlsxFiles) {
      try {
        const formData = new FormData();
        formData.append('sample_file', xlsxTemplateFile);
        formData.append('submission_file', file);
  
        const response = await api.post('/exams/grade-xlsx', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
  
        // Gọi API lấy thông tin sinh viên
        if (response.data.student_id) {
          try {
            const studentRes = await api.get(`/students/${response.data.student_id}`);
            response.data.student_name = studentRes.data.name;
          } catch (err) {
            console.error(`Error fetching student info for ${response.data.student_id}:`, err);
            response.data.student_name = 'Unknown';
          }
        }
  
        // Cập nhật kết quả ngay lập tức
        setExcelGradingResults(prev => [...prev, response.data]);
  
        toast({
          title: `Graded: ${file.name}`,
          description: `Score: ${response.data.score}`,
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
  
      } catch (error) {
        console.error(`Error grading ${file.name}:`, error);
        toast({
          title: `Grading failed for ${file.name}`,
          status: 'error',
          duration: 4000,
          isClosable: true,
        });
      }
    }
  
    setIsUploading(false);
  };

  
  function highlightDiff(text: string, highlights: string[], color: string) {
    if (!text || !highlights?.length) return text;

    const normalizedHighlights = highlights.map(word =>
      word.replace(/[.,!?;:"“”]/g, "").toLowerCase()
    );
  
    const tokens = text.split(/(\s+)/); // giữ khoảng trắng
    return tokens.map((word, idx) => {
      const cleaned = word.replace(/[.,!?;:"“”]/g, "").toLowerCase();
      if (normalizedHighlights.includes(cleaned)) {
        return (
          <Box as="span" key={idx} bg={color} px={1} borderRadius="md">
            {word}
          </Box>
        );
      }
      return word;
    });
  }

  
  
  
  return (
    <Box>
      <Heading size="lg" mb={4}>Exam Grading</Heading>
      
      <Tabs variant="enclosed" colorScheme="blue">
        <TabList>
          <Tab>DOCX Grading</Tab>
          <Tab>Excel Grading</Tab>
        </TabList>
        
        <TabPanels>
          {/* DOCX Grading Tab */}
          <TabPanel>
            <Card mb={6}>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <Heading size="md">Grade DOCX Exams</Heading>
                  <Text>Upload a student answer file (DOCX) and the exam template file (DOCX) to grade automatically.</Text>
                  
                  <FormControl>
                    <FormLabel>Student Answer File (DOCX)</FormLabel>
                    <HStack>
                      <Input
                        type="file"
                        accept=".docx"
                        multiple
                        onChange={handleDocxFileChange}
                        display="none"
                        ref={docxFileInputRef}
                      />
                      <Button 
                        leftIcon={<Icon as={FiUpload} />}
                        onClick={() => docxFileInputRef.current?.click()}
                        colorScheme={docxFiles.length > 0 ? "green" : "blue"}
                      >
                        {docxFiles.length > 0 ? `${docxFiles.length} file(s) selected` : 'Select Files'}
                      </Button>
                    </HStack>
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Exam Template File (DOCX)</FormLabel>
                    <HStack>
                      <Input
                        type="file"
                        accept=".docx"
                        onChange={handleExamTemplateChange}
                        display="none"
                        ref={examTemplateFileInputRef}
                      />
                      <Button 
                        leftIcon={<Icon as={FiUpload} />}
                        onClick={() => examTemplateFileInputRef.current?.click()}
                        colorScheme={examTemplateFile ? "green" : "blue"}
                      >
                        {examTemplateFile ? 'File Selected' : 'Select File'}
                      </Button>
                      {examTemplateFile && (
                        <Text noOfLines={1} flex="1">
                          {examTemplateFile.name}
                        </Text>
                      )}
                    </HStack>
                  </FormControl>
                  
                  <Button
                    colorScheme="blue"
                    isLoading={isUploading}
                    loadingText="Processing..."
                    leftIcon={<Icon as={FiCheck} />}
                    onClick={handleDocxSubmit}
                    isDisabled={!docxFiles.length || !examTemplateFile}
                  >
                    Submit for Grading
                  </Button>
                  
                  {isUploading && (
                    <Box mt={4}>
                      <Text mb={2}>Processing files...</Text>
                      <Progress size="xs" isIndeterminate colorScheme="blue" />
                    </Box>
                  )}
                </VStack>
              </CardBody>
            </Card>
            
            {/* Results Section */}
            {docxGradingResultsList.length > 0 && (
              <VStack spacing={6} mt={6} align="stretch">
                {docxGradingResultsList.map((result, idx) => (
                  <Card key={idx}>
                    <CardBody>
                      <VStack spacing={2} align="stretch">
                        <Heading size="sm">Result #{idx + 1}</Heading>
                        <Text><strong>Student:</strong> {result.student_name}</Text>
                        <Text><strong>Exam Code:</strong> {result.exam_code}</Text>
                        <Text><strong>Score:</strong> <span style={{ color: 'green' }}>{result.score}</span></Text>
                        <Box whiteSpace="pre-wrap" mt={4} p={4} bg="gray.50" borderRadius="md" fontSize="sm">
                          <Text fontWeight="bold" color="blue.600" mb={2}>Feedback Report</Text>
                          {(() => {
                            const parsedReport = parseReport(result.report);
                            if (!parsedReport) return <Text color="red.500">Invalid report format</Text>;

                            return (
                              <>
                                <Text color={parsedReport.overall_match ? "green.600" : "red.600"} fontWeight="semibold">
                                  Overall Result: {parsedReport.overall_match ? "PASS" : "FAIL"}
                                </Text>

                                {/* Formatting Evaluation */}
                                <Box mt={3}>
                                  <Text fontWeight="bold" color="purple.600">Formatting Evaluation</Text>

                                  {Object.entries(parsedReport.formatting || {}).map(([section, detail]: any, i) => (
                                    <Box key={i} pl={4} mt={2}>
                                      <Text fontWeight="medium" color={detail.matches ? "green.500" : "red.500"}>
                                        {section.replace(/_/g, ' ').toUpperCase()}: {detail.matches ? "MATCH" : "MISMATCH"}
                                      </Text>

                                      {/* Hiển thị similarity score nếu có */}
                                      {detail.similarity_score !== undefined && (
                                        <Text fontSize="sm" color="blue.500">Similarity Score: {detail.similarity_score}</Text>
                                      )}

                                      {/* Nếu có differences thì dùng accordion */}
                                      {!detail.matches && detail.differences && (
                                        <Accordion allowMultiple mt={2}>
                                          {Object.entries(detail.differences).map(([key, value]: any, idx) => (
                                            <AccordionItem key={idx}>
                                              <AccordionButton _expanded={{ bg: "gray.100" }}>
                                                <Box flex="1" textAlign="left">
                                                  <Text fontWeight="medium">{key.replace(/_/g, ' ')}</Text>
                                                </Box>
                                                <AccordionIcon />
                                              </AccordionButton>
                                              <AccordionPanel pb={4} fontSize="sm" color="gray.700">
                                                {typeof value === 'object' && !Array.isArray(value) ? (
                                                  <Box pl={2}>
                                                    {Object.entries(value).map(([subKey, subVal], subIdx) => (
                                                      <Text key={subIdx}>
                                                        • <strong>{subKey.replace(/_/g, ' ')}:</strong>{" "}
                                                        {typeof subVal === "object" && subVal !== null && "expected" in subVal && "actual" in subVal ? (
                                                          <>Expected: {subVal.expected}, Actual: {subVal.actual}</>
                                                        ) : (
                                                          String(subVal)
                                                        )}
                                                      </Text>
                                                    ))}
                                                  </Box>
                                                ) : (
                                                  <Text>{JSON.stringify(value)}</Text>
                                                )}
                                              </AccordionPanel>
                                            </AccordionItem>
                                          ))}
                                        </Accordion>
                                      )}
                                    </Box>
                                  ))}
                                </Box>

                                {/* Content Evaluation */}
                                <Box mt={4}>
                                  <Text fontWeight="bold" color="purple.600">Content Evaluation</Text>
                                  <Text fontWeight="medium" color={parsedReport.content?.matches ? "green.500" : "red.500"}>
                                    Content: {parsedReport.content?.matches ? "MATCH" : "MISMATCH"}
                                  </Text>
                                  {!parsedReport.content?.matches && (
                                    <Box pl={4} mt={2} color="gray.700">
                                      {parsedReport.content.differences?.paragraph_count && (
                                        <>
                                          <Text>Paragraph count:</Text>
                                          <Text pl={2}>Expected: {parsedReport.content.differences.paragraph_count.expected}</Text>
                                          <Text pl={2}>Actual: {parsedReport.content.differences.paragraph_count.actual}</Text>
                                        </>
                                      )}
                                      {Array.isArray(parsedReport.content.differences?.errors) &&
                                        parsedReport.content.differences.errors.map((err: any, idx: number) => (
                                          <Box key={idx} mt={4} borderLeft="3px solid" borderColor="gray.300" pl={3}>
                                            <Text fontWeight="semibold" color="orange.500">
                                              {err.type.replace(/_/g, " ")}
                                            </Text>

                                            {/* Hiển thị điểm similarity nếu có */}
                                            {err.similarity && (
                                              <Text fontSize="sm" color="gray.600">
                                                Similarity Score: <strong>{err.similarity}</strong>
                                              </Text>
                                            )}

                                            {err.type === "partially_similar_content" && (
                                              <>
                                                <Text mt={2} fontWeight="medium">Sample:</Text>
                                                <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">
                                                  {highlightDiff(err.sample_text, (err.diff?.removed || "").split(" "), "red.200")}
                                                </Box>

                                                <Text mt={2} fontWeight="medium">Submission:</Text>
                                                <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">
                                                  {highlightDiff(err.submission_text, (err.diff?.added || "").split(" "), "yellow.200")}
                                                </Box>
                                              </>
                                            )}

                                            {/* Các lỗi khác vẫn hiển thị bình thường */}
                                            {err.type !== "partially_similar_content" && (
                                              <Box mt={2}>
                                                {/* Thông báo chung nếu có */}
                                                {err.message && (
                                                  <Text color="red.500" fontWeight="medium">
                                                    {err.message}
                                                  </Text>
                                                )}

                                                {/* Lỗi font_difference hoặc tương tự */}
                                                {err.property && (
                                                  <Text>
                                                    <strong>Property:</strong> {err.property.replace(/_/g, " ")}
                                                  </Text>
                                                )}
                                                {err.expected !== undefined && err.actual !== undefined && (
                                                  <Text>
                                                    <strong>Expected:</strong> {String(err.expected)}{" "}
                                                    <strong>| Actual:</strong> {String(err.actual)}
                                                  </Text>
                                                )}

                                                {/* Hiển thị đoạn văn nếu có */}
                                                {err.text && (
                                                  <Text mt={1} fontStyle="italic" color="gray.700">
                                                    "{err.text}"
                                                  </Text>
                                                )}
                                                {err.sample_text && (
                                                  <>
                                                    <Text mt={2} fontWeight="medium">Sample:</Text>
                                                    <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">
                                                      {err.sample_text}
                                                    </Box>
                                                  </>
                                                )}
                                                {err.submission_text && (
                                                  <>
                                                    <Text mt={2} fontWeight="medium">Submission:</Text>
                                                    <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">
                                                      {err.submission_text}
                                                    </Box>
                                                  </>
                                                )}
                                              </Box>
                                            )}
                                          </Box>
                                      ))}
                                      <Text mt={2}>Similarity Score: {parsedReport.content.similarity_score}</Text>
                                    </Box>
                                  )}
                                </Box>

                                <Text mt={4} fontWeight="bold" color="blue.700">Final Score: {result.score}</Text>
                              </>
                            );
                          })()}
                        </Box>

                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </VStack>
            )}
          </TabPanel>
          
          {/* Excel Grading Tab (Left blank as requested) */}
          <TabPanel>
            <Card>
              <CardBody>
                <VStack spacing={4} align="stretch">
                  <Heading size="md">Excel Grading</Heading>
                  <FormControl>
                    <FormLabel>Student Excel Files (.xlsx)</FormLabel>
                    <HStack>
                      <Input
                        type="file"
                        accept=".xlsx"
                        multiple
                        onChange={(e) => setXlsxFiles(Array.from(e.target.files || []))}
                        display="none"
                        ref={xlsxFileInputRef}
                      />
                      <Button
                        leftIcon={<Icon as={FiUpload} />}
                        onClick={() => xlsxFileInputRef.current?.click()}
                        colorScheme={xlsxFiles.length > 0 ? "green" : "blue"}
                      >
                        {xlsxFiles.length > 0 ? `${xlsxFiles.length} file(s) selected` : 'Select Files'}
                      </Button>
                    </HStack>
                  </FormControl>
                  <FormControl>
                    <FormLabel>Exam Template Excel File (.xlsx)</FormLabel>
                    <HStack>
                      <Input
                        type="file"
                        accept=".xlsx"
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            setXlsxTemplateFile(e.target.files[0]);
                          }
                        }}
                        display="none"
                        ref={xlsxTemplateFileInputRef}
                      />
                      <Button
                        leftIcon={<Icon as={FiUpload} />}
                        onClick={() => xlsxTemplateFileInputRef.current?.click()}
                        colorScheme={xlsxTemplateFile ? "green" : "blue"}
                      >
                        {xlsxTemplateFile ? 'File Selected' : 'Select File'}
                      </Button>
                      {xlsxTemplateFile && (
                        <Text noOfLines={1} flex="1">
                          {xlsxTemplateFile.name}
                        </Text>
                      )}
                    </HStack>
                  </FormControl>
                  <Button colorScheme="blue" onClick={handleXlsxSubmit} isLoading={isUploading}>Submit for Grading</Button>
                </VStack>
              </CardBody>
            </Card>
            {excelGradingResults.length > 0 && (
              <VStack spacing={6} mt={6} align="stretch">
                {excelGradingResults.map((result, idx) => (
                  <Card key={idx}>
                    <CardBody>
                      <VStack spacing={2} align="stretch">
                        <Heading size="sm">Result #{idx + 1}</Heading>
                        <Text><strong>Student:</strong> {result.student_name || 'Unknown'}</Text>
                        <Text><strong>Exam Code:</strong> {result.exam_code || 'N/A'}</Text>
                        <Text><strong>Score:</strong> <span style={{ color: 'green' }}>{result.score}</span></Text>
                        <Box mt={4}
                          p={4}
                          bg="gray.50"
                          borderRadius="md"
                          fontSize="sm"
                          maxHeight="400px"
                          overflowY="auto"
                          border="1px solid"
                          borderColor="gray.200">
                          <Text fontWeight="bold" color="blue.600" mb={2}>Feedback Report</Text>

                          {(() => {
                            const parsedReport = typeof result.report === 'string' ? JSON.parse(result.report) : result.report;
                            if (!parsedReport) return <Text color="red.500">Invalid report format</Text>;

                            return (
                              <VStack align="start" spacing={2}>
                                <Text><strong>Student Name:</strong> {parsedReport.info_student?.name}</Text>
                                <Text><strong>Student Code:</strong> {parsedReport.info_student?.student_code}</Text>
                                <Text><strong>Exam Code:</strong> {parsedReport.info_student?.exam_code}</Text>
                                <Text><strong>Final Score:</strong> <span style={{ color: 'green' }}>{parsedReport.score}</span></Text>

                                <Box>
                                  <Text fontWeight="medium" color="purple.700">Penalties:</Text>
                                  {Object.entries(parsedReport.penalties || {}).map(([key, val]) => (
                                    <Text key={key} pl={2}>• {key.replace(/_/g, ' ')}: {String(val)}</Text>
                                  ))}
                                </Box>

                                <Box>
                                  <Text fontWeight="medium" color="purple.700">Cell Comparison:</Text>
                                  <Text>Matched Cells: {parsedReport.matched_cells} / {parsedReport.total_cells}</Text>
                                  <Text>Wrong Formula: {parsedReport.wrong_formula} / {parsedReport.total_formula}</Text>
                                </Box>

                                {Array.isArray(parsedReport.cell_errors) && parsedReport.cell_errors.length > 0 && (
                                  <Box mt={3} w="100%">
                                    <Text fontWeight="bold" color="red.500" mb={2}>Cell Errors</Text>
                                    <Table variant="simple" size="sm">
                                      <Thead>
                                        <Tr>
                                          <Th>Table</Th>
                                          <Th>Cell</Th>
                                          <Th>Errors</Th>
                                          <Th>Expected</Th>
                                          <Th>Actual</Th>
                                        </Tr>
                                      </Thead>
                                      <Tbody>
                                        {parsedReport.cell_errors.map((error: any, idx: number) => (
                                          <Tr key={idx}>
                                            <Td>{error.table_id}</Td>
                                            <Td>{error.coordinate}</Td>
                                            <Td>
                                              {error.errors?.map((e: string, i: number) => (
                                                <Badge key={i} colorScheme="red" mr={1}>{e}</Badge>
                                              ))}
                                              {error.error === 'missing' && (
                                                <Badge colorScheme="red">missing</Badge>
                                              )}
                                            </Td>
                                            <Td whiteSpace="pre-wrap">
                                              {JSON.stringify(error.expected, null, 2)}
                                            </Td>
                                            <Td whiteSpace="pre-wrap">
                                              {JSON.stringify(error.actual, null, 2)}
                                            </Td>
                                          </Tr>
                                        ))}
                                      </Tbody>
                                    </Table>
                                  </Box>
                                )}
                              </VStack>
                            );
                          })()}
                        </Box>

                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </VStack>
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
}
