import React, { useEffect, useState } from "react";
import { ScrollView, StatusBar, FlatList, Text, ActivityIndicator, View, TextInput, TouchableOpacity } from "react-native";
import styled from "styled-components/native";
import { Picker } from "@react-native-picker/picker";

import SampleCard, { SampleRow } from "../components/SampleCard";
import ClassificationPie from "../components/ClassificationPie";

/* ================================
   STYLED COMPONENTS
================================ */
const Root = styled.SafeAreaView`
  flex: 1;
  background: #1f1f1f;
  padding-top: 40px;
`;

const Header = styled.View`
  padding: 20px 16px 10px 16px;
  align-items: center;
`;

const Logo = styled.Image`
  width: 84px;
  height: 84px;
  margin-bottom: 8px;
`;

const Title = styled.Text`
  color: #ebe1bd;
  font-size: 22px;
  font-weight: bold;
`;

const Subtitle = styled.Text`
  color: #c3c8c7;
  font-size: 14px;
  margin-top: 2px;
`;

const Section = styled.View`
  margin: 18px 16px;
`;

const Input = styled.TextInput`
  background-color: #262626;
  color: #ebe1bd;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 8px;
`;

const Button = styled.TouchableOpacity`
  background-color: #3a4946;
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 10px;
  align-items: center;
`;

const ButtonText = styled.Text`
  color: #ebe1bd;
  font-weight: bold;
`;

const DropdownWrapper = styled.View`
  background-color: #262626;
  border-radius: 8px;
  margin-bottom: 12px;
`;

/* ================================
   MAIN COMPONENT
================================ */
export default function Dashboard() {
  type Batch = { id: number; name?: string; createdAt?: string };

  const API = "http://192.168.1.22:4000";

  const [batches, setBatches] = useState<Batch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<number | null>(null);

  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [stats, setStats] = useState<any>(null);

  const [newBatchName, setNewBatchName] = useState("");
  const [editBatchName, setEditBatchName] = useState("");

  const [loadingSamples, setLoadingSamples] = useState(false);
  const [selectedSamples, setSelectedSamples] = useState<number[]>([]);

  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [editSample, setEditSample] = useState<SampleRow | null>(null);
  const [editModalVisible, setEditModalVisible] = useState(false);

  /* ================================
     FETCH BATCHES
  ================================= */
  useEffect(() => {
    fetchBatches();
  }, []);

  const fetchBatches = async () => {
    try {
      const res = await fetch(`${API}/api/batches`);
      const json = await res.json();
      setBatches(json.batches ?? []);
    } catch (err) {
      console.error("Fetch batches error:", err);
    }
  };

  /* ================================
     FETCH SAMPLES + STATS
  ================================= */
  const fetchSamplesAndStats = async (batchId: number) => {
    setLoadingSamples(true);

    try {
      const [sRes, stRes] = await Promise.all([
        fetch(`${API}/api/batches/${batchId}/samples`),
        fetch(`${API}/api/batches/${batchId}/stats`),
      ]);

      const samplesJson = await sRes.json();
      const statsJson = await stRes.json();

      setSamples(samplesJson.samples ?? []);
      setStats(statsJson ?? null);

      const match = batches.find((b) => b.id === batchId);
      setEditBatchName(match?.name ?? "");
    } catch (err) {
      console.error("Fetch samples/stats error:", err);
    } finally {
      setLoadingSamples(false);
    }
  };

  /* ================================
     BATCH CRUD
  ================================= */
  const handleAddBatch = async () => {
    if (!newBatchName.trim()) return;

    const res = await fetch(`${API}/api/batches`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newBatchName }),
    });

    const json = await res.json();
    if (res.ok) {
      setBatches((prev) => [json.batch, ...prev]);
      setNewBatchName("");
    } else {
      alert("Failed to add batch");
    }
  };

  const handleUpdateBatch = async () => {
    if (!selectedBatch) return;

    const res = await fetch(`${API}/api/batches/${selectedBatch}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: editBatchName }),
    });

    if (res.ok) {
      fetchBatches();
      alert("Batch updated");
    } else {
      alert("Update failed");
    }
  };

  const handleDeleteBatch = async () => {
    if (!selectedBatch) return;

    const res = await fetch(`${API}/api/batches/${selectedBatch}`, {
      method: "DELETE",
    });

    if (res.ok) {
      setBatches((prev) => prev.filter((b) => b.id !== selectedBatch));
      setSelectedBatch(null);
      setSamples([]);
      setStats(null);
      alert("Batch deleted");
    }
  };

  const toggleSelect = (id: number) => {
    setSelectedIds(prev =>
      prev.includes(id)
        ? prev.filter(x => x !== id)
        : [...prev, id]
    );
  };

  const deleteSelected = async () => {
    if (selectedIds.length === 0) return;

    try {
      await fetch(`${API}/samples/deleteMany`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedIds }),
      });

      // Refresh UI
      setSamples(prev => prev.filter(s => !selectedIds.includes(s.id)));
      setSelectedIds([]);
    } catch (err) {
      console.error("Delete error:", err);
    }
  };

  const startEditing = (sample: SampleRow) => {
    setEditSample(sample);
    setEditModalVisible(true);
  };


  /* ================================
     RENDER
  ================================= */
  return (
    <Root>
      <StatusBar barStyle="light-content" />

      {/* HEADER */}
      <Header>
        <Logo source={require("../../assets/splash-logo.png")} />
        <Title>Fiber Optics Dashboard</Title>
        <Subtitle>Inspector View</Subtitle>
      </Header>

      <ScrollView contentContainerStyle={{ paddingBottom: 40 }}>
        {/* ========== BATCH MANAGEMENT ========== */}
        <Section>
          <Input
            placeholder="New Batch Name"
            placeholderTextColor="#666"
            value={newBatchName}
            onChangeText={setNewBatchName}
          />
          <Button onPress={handleAddBatch}>
            <ButtonText>Add Batch</ButtonText>
          </Button>

          <Button onPress={fetchBatches}>
            <ButtonText>Reload Batches</ButtonText>
          </Button>
        </Section>

        {/* ========== BATCH DROPDOWN ========== */}
        <Section>
          <DropdownWrapper>
            <Picker
              selectedValue={selectedBatch}
              onValueChange={(v) => {
                setSelectedBatch(v);
                if (v) fetchSamplesAndStats(v);
              }}
              dropdownIconColor="#EBE1BD"
              style={{ color: "#EBE1BD" }}
            >
              <Picker.Item label="Select Batch" value={null} />

              {batches.map((b) => (
                <Picker.Item
                  key={b.id}
                  label={`Batch #${b.id}${b.name ? " — " + b.name : ""}`}
                  value={b.id}
                />
              ))}
            </Picker>
          </DropdownWrapper>

          {/* Edit batch name */}
          {selectedBatch && (
            <>
              <Input
                placeholder="Edit Batch Name"
                placeholderTextColor="#666"
                value={editBatchName}
                onChangeText={setEditBatchName}
              />

              <Button onPress={handleUpdateBatch}>
                <ButtonText>Update Batch</ButtonText>
              </Button>

              <Button onPress={handleDeleteBatch}>
                <ButtonText>Delete Batch</ButtonText>
              </Button>
            </>
          )}
        </Section>

        {/* ========== PIE CHART ABOVE SAMPLE LIST ========== */}
        {samples.length > 0 && (
          <Section>
            <ClassificationPie samples={samples} />
          </Section>
        )}

        {/* ========== SAMPLE LIST ========== */}
        <Section>
          <Text
            style={{ color: "#EBE1BD", fontWeight: "bold", marginBottom: 10 }}
          >
            Samples
          </Text>

          {selectedIds.length > 0 && (
            <TouchableOpacity
              onPress={deleteSelected}
              style={{
                backgroundColor: "#7A2E2E",
                padding: 10,
                borderRadius: 8,
                marginBottom: 12,
              }}
            >
              <Text style={{ color: "#EBE1BD", textAlign: "center", fontWeight: "600" }}>
                Delete Selected ({selectedIds.length})
              </Text>
            </TouchableOpacity>
          )}

          {loadingSamples ? (
            <ActivityIndicator size="large" color="#EBE1BD" />
          ) : (
            <FlatList
              data={samples}
              scrollEnabled={false}
              keyExtractor={(item) => String(item.id)}
              renderItem={({ item }) => (
                <SampleCard
                  sample={item}
                  selected={selectedSamples.includes(item.id)}
                  onSelect={toggleSelect}
                  onEdit={startEditing}
                />
              )}
              ListEmptyComponent={
                <Text style={{ color: "#C3C8C7" }}>No samples</Text>
              }
            />
          )}
        </Section>
      </ScrollView>

      {editModalVisible && editSample && (
        <View
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.6)",
            justifyContent: "center",
            alignItems: "center",
            padding: 20,
            zIndex: 9999,
          }}
        >
          <View
            style={{
              width: "90%",
              backgroundColor: "#262626",
              padding: 20,
              borderRadius: 12,
            }}
          >
            <Text style={{ color: "#EBE1BD", fontSize: 18, marginBottom: 8 }}>
              Edit Sample #{editSample.id}
            </Text>

            <Text style={{ color: "#C3C8C7" }}>Classification</Text>
            <TextInput
              value={editSample.classification}
              onChangeText={(v) =>
                setEditSample({ ...editSample, classification: v })
              }
              style={{
                backgroundColor: "#3A4946",
                color: "#EBE1BD",
                padding: 8,
                borderRadius: 8,
                marginBottom: 12,
              }}
            />

            <TouchableOpacity
              onPress={async () => {
                try {
                  await fetch(`${API}/samples/${editSample.id}`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(editSample),
                  });

                  // Update UI
                  setSamples(prev =>
                    prev.map(s => (s.id === editSample.id ? editSample : s))
                  );
                } catch (err) {
                  console.error("Edit failed:", err);
                }

                setEditModalVisible(false);
              }}
              style={{
                backgroundColor: "#3A4946",
                padding: 12,
                borderRadius: 8,
                marginTop: 10,
              }}
            >
              <Text style={{ color: "#EBE1BD", textAlign: "center", fontWeight: "600" }}>
                Save Changes
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => setEditModalVisible(false)}
              style={{ marginTop: 10 }}
            >
              <Text style={{ color: "#C3C8C7", textAlign: "center" }}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

    </Root>
  );
}