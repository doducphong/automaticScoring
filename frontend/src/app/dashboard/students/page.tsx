'use client';

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
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
} from '@chakra-ui/react';
import { FiEdit, FiTrash2, FiPlus, FiSearch } from 'react-icons/fi';
import api from '@/lib/api';

interface Student {
  id: number;
  name: string;
  student_code: string;
}

export default function StudentsPage() {
  const [students, setStudents] = useState<Student[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentStudent, setCurrentStudent] = useState<Student | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [formData, setFormData] = useState({
    name: '',
    student_code: '',
  });
  const {
    isOpen: isAlertOpen,
    onOpen: onAlertOpen,
    onClose: onAlertClose,
  } = useDisclosure();
  const cancelRef = useRef(null);
  const [studentToDelete, setStudentToDelete] = useState<Student | null>(null);

  const toast = useToast();

  // Fetch students data
  useEffect(() => {
    const fetchStudents = async () => {
      try {
        const res = await api.get('/students');
      setStudents(res.data);
    } catch (error) {
      console.error('Error fetching students:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch students',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  fetchStudents();
}, [toast]);

  // Filter students based on search query
  const filteredStudents = students.filter(student => 
    student.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    student.student_code.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Handle opening the form for creating/editing student
  const handleOpenForm = (student: Student | null = null) => {
    if (student) {
      setCurrentStudent(student);
      setFormData({
        name: student.name,
        student_code: student.student_code,
      });
    } else {
      setCurrentStudent(null);
      setFormData({
        name: '',
        student_code: '',
      });
    }
    onOpen();
  };


  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  // Handle form submission
  const handleSubmit = async () => {
    try {
      if (currentStudent) {
        const res = await api.put(`/students/${currentStudent.id}`, formData);
        const updatedStudents = students.map(student =>
          student.id === currentStudent.id ? res.data : student
        );
        setStudents(updatedStudents);
        toast({
          title: 'Success',
          description: 'Student updated successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      } else {
        const res = await api.post('/students', formData);
        setStudents([...students, res.data]);
        toast({
          title: 'Success',
          description: 'Student added successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      }
      onClose();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to save student',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleDeleteClick = (student: Student) => {
    setStudentToDelete(student);
    onAlertOpen();
  };

  // Handle student deletion
  const confirmDelete = async () => {
    if (!studentToDelete) return;
    try {
      await api.delete(`/students/${studentToDelete.id}`);
      setStudents(prev =>
        prev.filter(student => student.id !== studentToDelete.id)
      );
      toast({
        title: 'Success',
        description: 'Student deleted successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to delete student',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setStudentToDelete(null);
      onAlertClose();
    }
  };

  return (
    <Box>
      <Heading size="lg" mb={4}>Student Management</Heading>
      
      <Card mb={6}>
        <CardBody>
          <Flex justify="space-between" mb={4}>
            <Flex maxW="400px">
              <Input
                placeholder="Search students..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                mr={2}
              />
              <IconButton
                aria-label="Search"
                icon={<FiSearch />}
                colorScheme="blue"
              />
            </Flex>
            
            <Button
              leftIcon={<FiPlus />}
              colorScheme="green"
              onClick={() => handleOpenForm()}
            >
              Add Student
            </Button>
          </Flex>
          
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Name</Th>
                  <Th>Student ID</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {isLoading ? (
                  <Tr>
                    <Td colSpan={3} textAlign="center">Loading...</Td>
                  </Tr>
                ) : filteredStudents.length === 0 ? (
                  <Tr>
                    <Td colSpan={3} textAlign="center">No students found</Td>
                  </Tr>
                ) : (
                  filteredStudents.map(student => (
                    <Tr key={student.id}>
                      <Td>{student.name}</Td>
                      <Td>{student.student_code}</Td>
                      <Td>
                        <IconButton
                          aria-label="Edit"
                          icon={<FiEdit />}
                          size="sm"
                          mr={2}
                          onClick={() => handleOpenForm(student)}
                        />
                        <IconButton
                          aria-label="Delete"
                          icon={<FiTrash2 />}
                          size="sm"
                          colorScheme="red"
                          onClick={() => handleDeleteClick(student)}
                        />
                      </Td>
                    </Tr>
                  ))
                )}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>

      {/* Student Form Modal */}
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            {currentStudent ? 'Edit Student' : 'Add New Student'}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Stack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Full Name</FormLabel>
                <Input
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                />
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Student Code</FormLabel>
                <Input
                  name="student_code"
                  value={formData.student_code}
                  onChange={handleInputChange}
                />
              </FormControl>
              
            </Stack>
          </ModalBody>

          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>
              Cancel
            </Button>
            <Button 
              colorScheme="blue"
              onClick={handleSubmit}
              disabled={!formData.name || !formData.student_code}
            >
              {currentStudent ? 'Update' : 'Create'}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
      {/* AlertDialog Confirm Delete */}
      <AlertDialog
        isOpen={isAlertOpen}
        leastDestructiveRef={cancelRef}
        onClose={onAlertClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Student
            </AlertDialogHeader>
            <AlertDialogBody>
              Are you sure you want to delete <strong>{studentToDelete?.name}</strong>?<br />
              This will also remove all related exams.
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onAlertClose}>Cancel</Button>
              <Button colorScheme="red" onClick={confirmDelete} ml={3}>Delete</Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Box>
  );
}
