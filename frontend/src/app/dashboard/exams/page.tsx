'use client';

import * as XLSX from 'xlsx';

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  Flex,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Input,
  IconButton,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  FormControl,
  FormLabel,
  Stack,
  useToast,
  Select,
  Badge,
  Text,
  HStack,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
  Tabs,
  TabList,
  TabPanels,
  TabPanel,
  Tab,
  VStack,
  Tooltip,
} from '@chakra-ui/react';
import { FiEdit, FiTrash2, FiPlus, FiSearch, FiUpload, FiDownload, FiEye } from 'react-icons/fi';
import { FaFileExcel } from 'react-icons/fa';
import api from '@/lib/api';

interface Exam {
  id: number;
  name: string;
  file_format: 'docx' | 'xlsx';
  exam_code: string;
  created_at: string;
  student_name: string;
  student_code: string;
  teacher_name?: string;
  score?: number;
  submission_url?: string;
}

export default function ExamsPage() {
  const [exams, setExams] = useState<Exam[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedExam, setSelectedExam] = useState<Exam | null>(null);
  const [feedback, setFeedback] = useState<any>(null);
  const [currentExam, setCurrentExam] = useState<Exam | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { isOpen: isDetailOpen, onOpen: onDetailOpen, onClose: onDetailClose } = useDisclosure();
  const [formData, setFormData] = useState<{
    name: string;
    score: string
  }>({
    name: '',
    score: '',
  });

  const exportToExcel = (examsToExport: Exam[], fileType: 'docx' | 'xlsx') => {
    const worksheetData = examsToExport.map(exam => ({
      Name: exam.name,
      Score: exam.score ?? '',
      'File Format': exam.file_format.toUpperCase(),
      Student: `${exam.student_name} (${exam.student_code})`,
      Teacher: exam.teacher_name ?? '',
      'Created At': formatDate(exam.created_at),
    }));
  
    const worksheet = XLSX.utils.json_to_sheet(worksheetData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Exams');
  
    const fileName = fileType === 'docx' ? 'report_docx.xlsx' : 'report_xlsx.xlsx';
    XLSX.writeFile(workbook, fileName);
  };
  
  
  const [xlsxData, setXlsxData] = useState<any[][]>([]);
  const toast = useToast();

  const readXlsxFromUrl = async (url: string): Promise<any[][]> => {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    return XLSX.utils.sheet_to_json(worksheet, { header: 1 });
  };

  // Fetch exams data
  useEffect(() => {
    const fetchExams = async () => {
      try {
        const res = await fetch('/api/v1/exams', {
          credentials: 'include', 
        });
        const data = await res.json();
        console.log('API Response:', data);
        setExams(data);
      } catch (error) {
        console.error('Error fetching exams:', error);
        toast({
          title: 'Error',
          description: 'Failed to fetch exams',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchExams();
  }, [toast]);

  useEffect(() => {
    if (selectedExam?.file_format === 'xlsx' && selectedExam.submission_url) {
      readXlsxFromUrl(selectedExam.submission_url).then(setXlsxData).catch(console.error);
    }
  }, [selectedExam]);

  // Filter exams based on search query
  const filteredExams = Array.isArray(exams)
  ? exams.filter(exam =>
      (exam.name?.toLowerCase() ?? '').includes(searchQuery.toLowerCase()) ||
      (exam.student_name?.toLowerCase() ?? '').includes(searchQuery.toLowerCase()) ||
      (exam.file_format?.toLowerCase() ?? '').includes(searchQuery.toLowerCase())
    )
  : [];

  const handleViewDetail = async (exam: Exam) => {
    setSelectedExam(exam);
    onDetailOpen();
    try {
      const res = await fetch(`/api/v1/exams/${exam.id}/report`, { credentials: 'include' });
      const data = await res.json();
      setFeedback(data);
    } catch {
      toast({
        title: 'Error',
        description: 'Could not load report.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  
  // Handle opening the form for creating/editing exam
  const handleOpenForm = (exam: Exam) => {
    setCurrentExam(exam);
    setFormData({
      name: exam.name,
      score: exam.score?.toString() || '',
    });
    onOpen();
  };

  // Handle form input changes
  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };


  // Handle form submission
  const handleSubmit = async () => {
    if (!currentExam) return;
    try {
      const res = await fetch(`/api/v1/exams/${currentExam.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          score: parseFloat(formData.score),
        }),
      });
      if (!res.ok) throw new Error('Failed to update exam');

      const updated = await res.json();
      setExams(prev => prev.map(ex => (ex.id === updated.id ? { ...ex, ...updated } : ex)));

      toast({
        title: 'Success',
        description: 'Exam updated successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      onClose();
    } catch (err) {
      toast({
        title: 'Error',
        description: 'Failed to update exam',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  // Handle exam deletion
  const handleDelete = async (examId: number) => {
    try {
      const res = await fetch(`/api/v1/exams/${examId}`, {
        method: 'DELETE',
      });
  
      if (!res.ok) throw new Error('Failed to delete exam');
  
      setExams(prev => prev.filter(exam => exam.id !== examId));
  
      toast({
        title: 'Success',
        description: 'Exam deleted successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (err) {
      toast({
        title: 'Error',
        description: 'Failed to delete exam',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      console.error('Error deleting exam:', err);
    }
  };

  // Format date
  const formatDate = (dateString: string) => new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  });

  // Handle download template file
  // const handleDownload = (exam: Exam) => {
  //   // In a real app, this would trigger a download from your backend
  //   toast({
  //     title: 'Download Started',
  //     description: `Downloading ${exam.template_file}`,
  //     status: 'info',
  //     duration: 3000,
  //     isClosable: true,
  //   });
  // };

  function parseReport(report: any) {
    if (!report || typeof report !== 'object') return null;
    return report;
  }
  
  function highlightDiff(text: string, diffWords: string[], bgColor: string) {
    if (!text) return null;
    const words = text.split(' ');
    return words.map((word, i) => {
      const key = `${word}-${i}`;
      if (diffWords.includes(word)) {
        return <Box as="span" key={key} bg={bgColor} px="1" mx="0.5">{word}</Box>;
      }
      return <Box as="span" key={key} mx="0.5">{word}</Box>;
    });
  }

  return (
    <Box>
      <Heading size="lg" mb={4}>Exam Management</Heading>

      <Tabs variant="enclosed" colorScheme="blue">
      <TabList>
        <Tab>DOCX Exams</Tab>
        <Tab>XLSX Exams</Tab>
      </TabList>

      <TabPanels>
        <TabPanel>
          {/* DOCX Exam Table */}
          {(() => {
            const docxExams = filteredExams.filter(exam => exam.file_format === 'docx');
            return (
                <Card mb={6}>
                  <CardBody>
                    <Flex justify="space-between" mb={4}>
                      <Flex maxW="400px">
                        <Input
                          placeholder="Search exams..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          mr={2}
                        />
                        <IconButton aria-label="Search" icon={<FiSearch />} colorScheme="blue" />
                      </Flex>
                      <Tooltip label="Export DOCX Exams to Excel">
                        <IconButton
                          aria-label="Export DOCX to Excel"
                          icon={<FaFileExcel />}
                          colorScheme="green"
                          onClick={() => exportToExcel(docxExams, 'docx')}
                        />
                      </Tooltip>
                    </Flex>

                    <Box overflowX="auto">
                      <Table variant="simple">
                        <Thead>
                          <Tr>
                            <Th>Name</Th>
                            <Th>Score</Th>
                            <Th>File Format</Th>
                            <Th>Student</Th>
                            <Th>Teacher</Th>
                            <Th>Created At</Th>
                            <Th>Actions</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {isLoading ? (
                            <Tr>
                              <Td colSpan={7} textAlign="center">Loading...</Td>
                            </Tr>
                          ) : filteredExams.length === 0 ? (
                            <Tr>
                              <Td colSpan={7} textAlign="center">No exams found</Td>
                            </Tr>
                          ) : (
                            docxExams.map(exam => (
                              <Tr key={exam.id}>
                                <Td>{exam.name}</Td>
                                <Td>{exam.score ?? '—'}</Td>
                                <Td>
                                  <Badge colorScheme={exam.file_format === 'docx' ? 'blue' : 'purple'}>
                                    {exam.file_format.toUpperCase()}
                                  </Badge>
                                </Td>
                                <Td>{exam.student_name} ({exam.student_code})</Td>
                                <Td>{exam.teacher_name ?? '—'}</Td>
                                <Td>{formatDate(exam.created_at)}</Td>
                                <Td>
                                  <HStack spacing={2}>
                                    <IconButton
                                      aria-label="Edit"
                                      icon={<FiEdit />}
                                      size="sm"
                                      onClick={() => handleOpenForm(exam)}
                                    />
                                    <IconButton
                                      aria-label="Delete"
                                      icon={<FiTrash2 />}
                                      size="sm"
                                      colorScheme="red"
                                      onClick={() => handleDelete(exam.id)}
                                    />
                                    {exam.submission_url && (
                                      <IconButton
                                        aria-label="Download"
                                        icon={<FiDownload />}
                                        size="sm"
                                        colorScheme="green"
                                        as="a"
                                        href={exam.submission_url}
                                        target="_blank"
                                        download
                                      />
                                    )}
                                    <IconButton aria-label="View Detail" icon={<FiEye />} size="sm" onClick={() => handleViewDetail(exam)} />
                                  </HStack>
                                </Td>
                              </Tr>
                            ))
                          )}
                        </Tbody>
                      </Table>
                    </Box>
                  </CardBody>
                </Card>
            );
          })()}
        </TabPanel>

        <TabPanel>
          {/* XLSX Exam Table */}
          {(() => {
            const xlsxExams = filteredExams.filter(exam => exam.file_format === 'xlsx');
            return (
              <Card mb={6}>
                <CardBody>
                  <Flex justify="space-between" mb={4}>
                    <Flex maxW="400px">
                      <Input
                        placeholder="Search exams..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        mr={2}
                      />
                      <IconButton aria-label="Search" icon={<FiSearch />} colorScheme="blue" />
                    </Flex>
                    <Tooltip label="Export XLSX Exams to Excel">
                      <IconButton
                        aria-label="Export XLSX to Excel"
                        icon={<FaFileExcel />}
                        colorScheme="green"
                        onClick={() => exportToExcel(xlsxExams, 'xlsx')} 
                      />
                    </Tooltip>
                  </Flex>

                  <Box overflowX="auto">
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Name</Th>
                          <Th>Score</Th>
                          <Th>File Format</Th>
                          <Th>Student</Th>
                          <Th>Teacher</Th>
                          <Th>Created At</Th>
                          <Th>Actions</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {isLoading ? (
                          <Tr>
                            <Td colSpan={7} textAlign="center">Loading...</Td>
                          </Tr>
                        ) : xlsxExams.length === 0 ? (
                          <Tr>
                            <Td colSpan={7} textAlign="center">No XLSX exams found</Td>
                          </Tr>
                        ) : (
                          xlsxExams.map(exam => (
                            <Tr key={exam.id}>
                              <Td>{exam.name}</Td>
                              <Td>{exam.score ?? '—'}</Td>
                              <Td><Badge colorScheme='purple'>XLSX</Badge></Td>
                              <Td>{exam.student_name} ({exam.student_code})</Td>
                              <Td>{exam.teacher_name ?? '—'}</Td>
                              <Td>{formatDate(exam.created_at)}</Td>
                              <Td>
                                <HStack spacing={2}>
                                  <IconButton aria-label="Edit" icon={<FiEdit />} size="sm" onClick={() => handleOpenForm(exam)} />
                                  <IconButton aria-label="Delete" icon={<FiTrash2 />} size="sm" colorScheme="red" onClick={() => handleDelete(exam.id)} />
                                  {exam.submission_url && (
                                    <IconButton aria-label="Download" icon={<FiDownload />} size="sm" colorScheme="green" as="a" href={exam.submission_url} target="_blank" download />
                                  )}
                                  <IconButton aria-label="View Detail" icon={<FiEye />} size="sm" onClick={() => handleViewDetail(exam)} />
                                </HStack>
                              </Td>
                            </Tr>
                          ))
                        )}
                      </Tbody>
                    </Table>
                  </Box>
                </CardBody>
              </Card>
            );
          })()}
        </TabPanel>
      </TabPanels>
    </Tabs>

      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Edit Exam</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Stack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Name</FormLabel>
                <Input
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                />
              </FormControl>
              <FormControl>
                <FormLabel>Score</FormLabel>
                <Input
                  name="score"
                  type="number"
                  step="0.01"
                  value={formData.score}
                  onChange={handleInputChange}
                />
              </FormControl>
            </Stack>
          </ModalBody>
          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>Cancel</Button>
            <Button colorScheme="blue" onClick={handleSubmit} disabled={!formData.name}>Update</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
      <Modal isOpen={isDetailOpen} onClose={onDetailClose} size="6xl">
        <ModalOverlay />
        <ModalContent maxW="90vw">
          <ModalHeader>Exam Detail</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Flex gap={4}>
              <Box flex="3" maxW="60%">
                <Text fontWeight="bold" mb={2}>Submission Preview</Text>
                {selectedExam?.submission_url ? (
                  selectedExam?.file_format === 'xlsx' ? (
                    <Box overflowX="auto" maxH="600px" border="1px solid #ccc" borderRadius="md" p={2}>
                      <Table variant="striped" size="sm">
                        <Tbody>
                          {xlsxData.map((row, rowIndex) => (
                            <Tr key={rowIndex}>
                              {row.map((cell, colIndex) => (
                                <Td key={colIndex}>{cell}</Td>
                              ))}
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    </Box>
                  ) : (
                    <iframe
                      src={`https://docs.google.com/gview?url=${encodeURIComponent(selectedExam.submission_url)}&embedded=true`}
                      style={{ width: '100%', height: '600px', border: '1px solid #ccc' }}
                      title="Submission Document"
                    />
                  )
                ) : (
                  <Text>No submission uploaded</Text>
                )}
              </Box>
              <Box flex="2" bg="gray.50" p={4} borderRadius="md" overflowY="auto" maxH="600px">
                <Text fontWeight="bold" color="blue.600" mb={2}>Feedback Report</Text>
                {(() => {
                  const parsedReport = parseReport(feedback?.report);
                  if (!parsedReport) return <Text color="red.500">Invalid report format</Text>;

                  if (selectedExam?.file_format === 'xlsx') {
                    return (
                      <VStack align="start" spacing={3} maxH="600px" overflowY="auto" pr={2}>
                        <Text color="blue.600" fontWeight="bold" fontSize="md">
                          Final Score: <Box as="span" color="green.600">{feedback?.score}</Box>
                        </Text>
      
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
                                    </Td>
                                    <Td whiteSpace="pre-wrap">{JSON.stringify(error.expected, null, 2)}</Td>
                                    <Td whiteSpace="pre-wrap">{JSON.stringify(error.actual, null, 2)}</Td>
                                  </Tr>
                                ))}
                              </Tbody>
                            </Table>
                          </Box>
                        )}
                      </VStack>
                    );
                  }
      

                  return (
                    <>
                      <Text color={parsedReport.overall_match ? "green.600" : "red.600"} fontWeight="semibold">
                        Overall Result: {parsedReport.overall_match ? "PASS" : "FAIL"}
                      </Text>

                      {/* --- Formatting --- */}
                      <Box mt={3}>
                        <Text fontWeight="bold" color="purple.600">Formatting Evaluation</Text>
                        {Object.entries(parsedReport.formatting || {}).map(([section, detail]: any, i) => (
                          <Box key={i} pl={4} mt={2}>
                            <Text fontWeight="medium" color={detail.matches ? "green.500" : "red.500"}>
                              {section.replace(/_/g, ' ').toUpperCase()}: {detail.matches ? "MATCH" : "MISMATCH"}
                            </Text>
                            {detail.similarity_score !== undefined && (
                              <Text fontSize="sm" color="blue.500">Similarity Score: {detail.similarity_score}</Text>
                            )}
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

                      {/* --- Content --- */}
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
                                  {err.type !== "partially_similar_content" && (
                                    <Box mt={2}>
                                      {err.message && (
                                        <Text color="red.500" fontWeight="medium">{err.message}</Text>
                                      )}
                                      {err.property && (
                                        <Text><strong>Property:</strong> {err.property.replace(/_/g, " ")}</Text>
                                      )}
                                      {err.expected !== undefined && err.actual !== undefined && (
                                        <Text><strong>Expected:</strong> {err.expected} | <strong>Actual:</strong> {err.actual}</Text>
                                      )}
                                      {err.text && (
                                        <Text mt={1} fontStyle="italic" color="gray.700">"{err.text}"</Text>
                                      )}
                                      {err.sample_text && (
                                        <>
                                          <Text mt={2} fontWeight="medium">Sample:</Text>
                                          <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">{err.sample_text}</Box>
                                        </>
                                      )}
                                      {err.submission_text && (
                                        <>
                                          <Text mt={2} fontWeight="medium">Submission:</Text>
                                          <Box bg="gray.50" p={2} borderRadius="md" whiteSpace="pre-wrap">{err.submission_text}</Box>
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

                      <Text mt={4} fontWeight="bold" color="blue.700">Final Score: {feedback.score}</Text>
                    </>
                  );
                })()}
              </Box>

            </Flex>
          </ModalBody>
          <ModalFooter>
            <Button onClick={onDetailClose}>Close</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
}
